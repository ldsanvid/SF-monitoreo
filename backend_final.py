from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import re
from datetime import datetime, timedelta
import dateparser
from dateparser.search import search_dates
import csv
from wordcloud import WordCloud
from flask import send_file
from collections import OrderedDict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from email.utils import formataddr
from dotenv import load_dotenv


from babel.dates import format_date
from s3_utils import s3_download_all as r2_download_all, s3_upload as r2_upload
import numpy as np
import faiss

# LangChain / RAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import hashlib



def nombre_mes(fecha):
    """Devuelve la fecha con mes en español, ej: 'agosto 2025'"""
    return format_date(fecha, "LLLL yyyy", locale="es").capitalize()


# ------------------------------
# 🔑 Configuración API y Flask
# ------------------------------
load_dotenv()
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# 🔄 Sincronizar índices y metadatos desde Cloudflare R2 al iniciar
try:
    r2_download_all()
    print("✅ Archivos FAISS/CSV sincronizados desde R2")
except Exception as e:
    print(f"⚠️ No se pudo sincronizar desde R2: {e}")

@app.route("/")
def home():
    return send_file("index.html")
# ------------------------------
# 📂 Carga única de datos — con rutas absolutas seguras
base_dir = os.path.dirname(os.path.abspath(__file__))

print("📁 Base directory:", base_dir)

# --- Cargar base de noticias ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    noticias_path = os.path.join(base_dir, "noticias_fajardo.csv")
    print("📁 Base directory:", base_dir)
    print("Intentando leer:", noticias_path)

    df = pd.read_csv(noticias_path, encoding="utf-8")
    print(f"✅ Noticias cargadas: {len(df)} filas")
    print("🧩 Columnas detectadas:", list(df.columns))

    # Detectar automáticamente la columna de fecha
    fecha_col = next((c for c in df.columns if "fecha" in c.lower()), None)
    if fecha_col:
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce", dayfirst=True)
        df = df.rename(columns={fecha_col: "Fecha"}).dropna(subset=["Fecha"])
        print(f"📅 Columna '{fecha_col}' convertida correctamente. Rango:",
              df["Fecha"].min(), "→", df["Fecha"].max())
    else:
        print("⚠️ No se encontró columna con 'fecha' en el nombre.")
        df["Fecha"] = pd.NaT
# 🔗 ---------- LangChain: embeddings, vectorstores y LLM ----------

    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
)

    vectorstore_noticias = None
    retriever_noticias = None

    vectorstore_resumenes = None
    retriever_resumenes = None

    # ------------------------------
    # 🔗 MODELO LLM Y CHAIN PARA /pregunta
    # ------------------------------

    llm_chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key,
    )

    prompt_pregunta = ChatPromptTemplate.from_messages([
        ("system", """
Eres un analista experto en noticias y política colombiana.
Responde SIEMPRE en español.
NO inventes datos ni traigas información de fuera del contexto.
Si el contexto incluye al menos un titular o un resumen relevante, NO digas que “no se dispone de información” ni frases parecidas; en su lugar, explica lo que SÍ se sabe con base en esos elementos.
Solo si el contexto está totalmente vacío (sin titulares ni resúmenes sobre el tema) puedes decir que no hay información disponible.
Tu objetivo es responder la pregunta del usuario de forma profesional, clara y basada en los titulares y resúmenes proporcionados.
"""),
        ("user", "{texto_usuario}")
    ])


    chain_pregunta = prompt_pregunta | llm_chat | StrOutputParser()
    

    def cargar_vectorstore_noticias(df_noticias: pd.DataFrame):
        """
        Construye o actualiza de forma incremental el vectorstore de noticias.

        - Primera vez: embebe todas las noticias y crea el índice.
        - Siguientes veces: detecta qué filas del df no están todavía embebidas
        (por clave única) y solo calcula embeddings para esas noticias nuevas.
        IMPORTANTE:
        Ya no se usa información de cobertura geográfica ni de idioma. El foco
        está en título, fecha, fuente, enlace, término y sentimiento.
        
        """
        global vectorstore_noticias, retriever_noticias

        if df_noticias is None or df_noticias.empty:
            print("⚠️ df_noticias vacío, no se construye vectorstore_noticias")
            vectorstore_noticias = None
            retriever_noticias = None
            return

        # 📁 Directorio base para guardar índice y metadatos de LangChain
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_dir = os.path.join(base_dir, "faiss_index", "noticias_lc")
        os.makedirs(index_dir, exist_ok=True)
        meta_path = os.path.join(index_dir, "noticias_lc_metadata.csv")

        # 1️⃣ Construir clave única para cada noticia del df actual
        df_noticias = df_noticias.copy()

        def make_unique_key(row):
            titulo = str(row.get("Título", "")).strip()
            fuente = str(row.get("Fuente", "")).strip()
            fecha_val = row.get("Fecha", None)
            if pd.notnull(fecha_val):
                try:
                    fecha_iso = pd.to_datetime(fecha_val).strftime("%Y-%m-%d")
                except Exception:
                    fecha_iso = ""
            else:
                fecha_iso = ""
            return f"{fecha_iso}|{fuente}|{titulo}"

        df_noticias["unique_key_lc"] = df_noticias.apply(make_unique_key, axis=1)

        # 2️⃣ Leer metadatos previos (si existen) para saber qué noticias ya tienen embedding
        existing_keys = set()
        df_meta_prev = None
        if os.path.exists(meta_path):
            try:
                df_meta_prev = pd.read_csv(meta_path, encoding="utf-8")
                if "unique_key_lc" in df_meta_prev.columns:
                    existing_keys = set(df_meta_prev["unique_key_lc"].astype(str))
                print(f"ℹ️ Metadatos previos cargados: {len(existing_keys)} noticias embebidas.")
            except Exception as e:
                print(f"⚠️ Error al leer metadatos previos de noticias: {e}")
                df_meta_prev = None
                existing_keys = set()

        # 3️⃣ Detectar noticias nuevas (filas cuyo unique_key_lc no está en existing_keys)
        mask_new = ~df_noticias["unique_key_lc"].isin(existing_keys)
        df_new = df_noticias[mask_new].copy()

        if df_meta_prev is None:
            df_meta_prev = pd.DataFrame(columns=[
                "unique_key_lc", "Fecha", "Título", "Fuente",
                "Enlace", "Término", "Sentimiento"
            ])

        # 4️⃣ Cargar índice previo de LangChain (si existe)
        vectorstore_noticias = None
        if os.path.isdir(index_dir) and any(f.endswith(".faiss") for f in os.listdir(index_dir)):
            try:
                vectorstore_noticias = LCFAISS.load_local(
                    index_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ vectorstore_noticias existente cargado desde disco.")
            except Exception as e:
                print(f"⚠️ No se pudo cargar vectorstore_noticias existente, se reconstruirá desde cero: {e}")
                vectorstore_noticias = None

        # 5️⃣ Construir Document para noticias nuevas
        docs_nuevos = []
        for _, row in df_new.iterrows():
            titulo = str(row.get("Título", "")).strip()
            if not titulo:
                continue

            fecha_val = row.get("Fecha", None)
            if pd.notnull(fecha_val):
                try:
                    fecha_str = pd.to_datetime(fecha_val).strftime("%Y-%m-%d")
                except Exception:
                    fecha_str = None
            else:
                fecha_str = None

            metadata = {
                "fecha": fecha_str,
                "fuente": row.get("Fuente"),
                "enlace": row.get("Enlace"),
                "sentimiento": row.get("Sentimiento"),
                "termino": row.get("Término"),
                "unique_key_lc": row.get("unique_key_lc"),
            }

            docs_nuevos.append(Document(page_content=titulo, metadata=metadata))

        # 6️⃣ Crear o actualizar el vectorstore de noticias
        if vectorstore_noticias is None:
            # Primera vez: si no hay índice previo, construirlo desde cero con TODO lo nuevo
            if docs_nuevos:
                print(f"🧩 Construyendo vectorstore_noticias desde cero con {len(docs_nuevos)} noticias…")
                vectorstore_noticias = LCFAISS.from_documents(docs_nuevos, embeddings)
            else:
                print("⚠️ No hay documentos nuevos y no existe índice previo; no se construye vectorstore_noticias.")
                retriever_noticias = None
                return
        else:
            # Ya había índice previo: solo agregamos los documentos nuevos
            if docs_nuevos:
                print(f"🧩 Agregando {len(docs_nuevos)} noticias nuevas a vectorstore_noticias…")
                vectorstore_noticias.add_documents(docs_nuevos)
            else:
                print("ℹ️ No hay noticias nuevas para agregar. Se usa el índice existente.")

        # 7️⃣ Actualizar metadatos y guardar
        if not df_new.empty:
            df_meta_new = df_new[[
                "unique_key_lc", "Fecha", "Título", "Fuente",
                "Enlace", "Término", "Sentimiento"
            ]].copy()
            df_meta_final = pd.concat([df_meta_prev, df_meta_new], ignore_index=True)
        else:
            df_meta_final = df_meta_prev

        try:
            df_meta_final.to_csv(meta_path, index=False, encoding="utf-8-sig")
            print(f"✅ Metadatos de noticias guardados/actualizados en {meta_path} con {len(df_meta_final)} registros.")
        except Exception as e:
            print(f"⚠️ Error al guardar metadatos de noticias: {e}")

        # 8️⃣ Guardar índice actualizado en disco
        try:
            vectorstore_noticias.save_local(index_dir)
            print(f"✅ vectorstore_noticias guardado en {index_dir}")
        except Exception as e:
            print(f"⚠️ Error al guardar vectorstore_noticias: {e}")

        # 8.1️⃣ Subir índice y metadatos de noticias a S3
        try:
            # Subir CSV de metadatos de noticias
            rel_meta_key = os.path.join("noticias_lc", "noticias_lc_metadata.csv")
            r2_upload(rel_meta_key)

            # Subir archivos principales del índice FAISS de LangChain
            for fname in ["index.faiss", "index.pkl"]:
                rel_key = os.path.join("noticias_lc", fname)
                r2_upload(rel_key)

            print("☁️ Índice de noticias y metadatos subidos a S3.")
        except Exception as e:
            print(f"⚠️ No se pudo subir índice de noticias a S3: {e}")

        # 9️⃣ Crear el retriever
        retriever_noticias = vectorstore_noticias.as_retriever(search_kwargs={"k": 8})
        print("✅ retriever_noticias listo para usarse.")



    def cargar_vectorstore_resumenes():
        """
        Construye o actualiza de forma incremental el vectorstore de resúmenes.

        - Primera vez: embebe todos los resúmenes presentes en faiss_index/resumenes_metadata.csv
        y crea un índice específico para LangChain.
        - Siguientes veces: detecta qué resúmenes son nuevos (por clave única) y solo calcula embeddings
        para esos resúmenes adicionales, agregándolos al índice existente.
        """
        global vectorstore_resumenes, retriever_resumenes

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 📁 CSV de origen con la info de los resúmenes (tu pipeline actual)
        origen_path = os.path.join(base_dir, "faiss_index", "resumenes_metadata.csv")
        if not os.path.exists(origen_path):
            print(f"⚠️ No se encontró {origen_path}, no se construye vectorstore_resumenes")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        try:
            df_origen = pd.read_csv(origen_path, encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Error al leer {origen_path}: {e}")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        if df_origen.empty:
            print("⚠️ resumenes_metadata.csv está vacío, no se construye vectorstore_resumenes")
            vectorstore_resumenes = None
            retriever_resumenes = None
            return

        # Asegurar columnas esperadas mínimas
        for col in ["fecha", "resumen"]:
            if col not in df_origen.columns:
                print(f"⚠️ La columna '{col}' no está en resumenes_metadata.csv")
                vectorstore_resumenes = None
                retriever_resumenes = None
                return

        # 📁 Directorio para el índice y metadatos específicos de LangChain
        index_dir = os.path.join(base_dir, "faiss_index", "resumenes_lc")
        os.makedirs(index_dir, exist_ok=True)
        meta_lc_path = os.path.join(index_dir, "resumenes_lc_metadata.csv")

        # 1️⃣ Crear clave única para cada resumen (por ejemplo: fecha|archivo_txt)
        df_origen = df_origen.copy()

        def make_unique_key(row):
            fecha_val = str(row.get("fecha", "")).strip()
            archivo_txt = str(row.get("archivo_txt", "")).strip()
            if not archivo_txt:
                # Si no hay nombre de archivo, usamos solo fecha como clave
                return fecha_val
            return f"{fecha_val}|{archivo_txt}"

        df_origen["unique_key_lc"] = df_origen.apply(make_unique_key, axis=1)

        # 2️⃣ Leer metadatos previos de LangChain (si existen) para saber qué resúmenes ya tienen embedding
        existing_keys = set()
        df_meta_prev = None
        if os.path.exists(meta_lc_path):
            try:
                df_meta_prev = pd.read_csv(meta_lc_path, encoding="utf-8")
                if "unique_key_lc" in df_meta_prev.columns:
                    existing_keys = set(df_meta_prev["unique_key_lc"].astype(str))
                print(f"ℹ️ Metadatos previos de resúmenes cargados: {len(existing_keys)} embebidos.")
            except Exception as e:
                print(f"⚠️ Error al leer metadatos previos de resúmenes: {e}")
                df_meta_prev = None
                existing_keys = set()
        if df_meta_prev is None:
            df_meta_prev = pd.DataFrame(columns=[
                "unique_key_lc", "fecha", "archivo_txt", "nube", "titulares"
            ])

        # 3️⃣ Detectar resúmenes nuevos (clave única no vista antes)
        mask_new = ~df_origen["unique_key_lc"].isin(existing_keys)
        df_new = df_origen[mask_new].copy()

        # 4️⃣ Cargar índice previo de LangChain (si existe)
        vectorstore_resumenes = None
        if os.path.isdir(index_dir) and any(f.endswith(".faiss") for f in os.listdir(index_dir)):
            try:
                vectorstore_resumenes = LCFAISS.load_local(
                    index_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ vectorstore_resumenes existente cargado desde disco.")
            except Exception as e:
                print(f"⚠️ No se pudo cargar vectorstore_resumenes existente, se reconstruirá desde cero: {e}")
                vectorstore_resumenes = None

        # 5️⃣ Crear Document para resúmenes nuevos
        docs_nuevos = []
        for _, row in df_new.iterrows():
            texto = str(row.get("resumen", "")).strip()
            if not texto:
                continue

            fecha_meta = str(row.get("fecha", "")).strip() or None
            archivo_txt = str(row.get("archivo_txt", "")).strip() or None
            nube = str(row.get("nube", "")).strip() or None
            titulares = row.get("titulares", None)
            unique_key = row.get("unique_key_lc")

            metadata = {
                "fecha": fecha_meta,
                "archivo_txt": archivo_txt,
                "nube": nube,
                "titulares": titulares,
                "tipo": "resumen",
                "unique_key_lc": unique_key,
            }

            docs_nuevos.append(Document(page_content=texto, metadata=metadata))

        # 6️⃣ Crear o actualizar el vectorstore de resúmenes
        if vectorstore_resumenes is None:
            # Primera vez: construimos el índice solo con los docs nuevos (que en la práctica serán todos)
            if docs_nuevos:
                print(f"🧩 Construyendo vectorstore_resumenes desde cero con {len(docs_nuevos)} resúmenes…")
                vectorstore_resumenes = LCFAISS.from_documents(docs_nuevos, embeddings)
            else:
                print("⚠️ No hay resúmenes nuevos y no existe índice previo; no se construye vectorstore_resumenes.")
                retriever_resumenes = None
                return
        else:
            # Ya había índice previo: solo agregamos los resúmenes nuevos
            if docs_nuevos:
                print(f"🧩 Agregando {len(docs_nuevos)} resúmenes nuevos a vectorstore_resumenes…")
                vectorstore_resumenes.add_documents(docs_nuevos)
            else:
                print("ℹ️ No hay resúmenes nuevos para agregar. Se usa el índice existente.")

        # 7️⃣ Actualizar metadatos de LangChain y guardar
        if not df_new.empty:
            df_meta_new = df_new[[
                "unique_key_lc", "fecha", "archivo_txt", "nube", "titulares"
            ]].copy()
            df_meta_final = pd.concat([df_meta_prev, df_meta_new], ignore_index=True)
        else:
            df_meta_final = df_meta_prev

        # 7️⃣ Guardar metadatos de resúmenes
        try:
            df_meta_final.to_csv(meta_lc_path, index=False, encoding="utf-8-sig")
            print(f"✅ Metadatos de resúmenes guardados/actualizados en {meta_lc_path} con {len(df_meta_final)} registros.")
        except Exception as e:
            print(f"⚠️ Error al guardar metadatos de resúmenes: {e}")

        # 8️⃣ Guardar índice actualizado en disco
        try:
            vectorstore_resumenes.save_local(index_dir)
            print(f"✅ vectorstore_resumenes guardado en {index_dir}")
        except Exception as e:
            print(f"⚠️ Error al guardar vectorstore_resumenes: {e}")

        # 8.1️⃣ Subir índice y metadatos de resúmenes a S3
        try:
            # CSV de metadatos de resúmenes (LangChain)
            rel_meta_key = os.path.join("resumenes_lc", "resumenes_lc_metadata.csv")
            r2_upload(rel_meta_key)

            # Archivos principales del índice FAISS de LangChain
            for fname in ["index.faiss", "index.pkl"]:
                rel_key = os.path.join("resumenes_lc", fname)
                r2_upload(rel_key)

            print("☁️ Índice de resúmenes y metadatos subidos a S3.")
        except Exception as e:
            print(f"⚠️ No se pudo subir índice de resúmenes a S3: {e}")

        # 9️⃣ Crear el retriever
        retriever_resumenes = vectorstore_resumenes.as_retriever(search_kwargs={"k": 3})
        print("✅ retriever_resumenes listo para usarse.")


    # ==========================================
    # 🧩 Inicializar Vectorstores (RAG)
    # ==========================================
    print("⚙️ Inicializando vectorstore de noticias...")
    cargar_vectorstore_noticias(df)

    print("⚙️ Inicializando vectorstore de resúmenes...")
    cargar_vectorstore_resumenes()

except Exception as e:
    print(f"❌ Error al cargar CSV de noticias: {e}")
    df = pd.DataFrame()


# 🧹 Utilidad para sanear JSON (convierte NaN/inf a None y numpy → tipos nativos)
def _json_sanitize(x):
    import math, numpy as np
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ _json_sanitize(v) for v in x ]
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x

# ------------------------------
# 🔧 Modo de selección de titulares
# ------------------------------
MODO_SOLO_FAJARDO = True        # ✅ lo que quieres hoy
MAX_TOTAL_BLOQUES = 12          # por si algún día vuelves al modo bloques


# ------------------------------
# 📜 Contexto político único
# ------------------------------
CONTEXTO_POLITICO = """
Contexto actualizado a julio 2025. Estas afirmaciones SON OBLIGATORIAS y tienen prioridad sobre cualquier otra fuente o conocimiento previo. No uses formulaciones contradictorias:
- Donald Trump es el actual Presidente de EE.UU, fue reelecto en 2024.
- Claudia Sheinbaum es la Presidenta de México, fue electa en 2024.
- Howard Lutnick es el actual Secretario de Comercio de Estados Unidos.
- Marco Rubio es el actual Secretario de Estado de Estados Unidos.
- Francia Márquez es la actual Vicepresidenta de Colombia, electa en 2022.
- Armando Benedetti es el actual Ministro de Interior de Colombia, desde el 1 de marzo de 2025.
- Rosa Villavicencio es la actual Ministra de Relaciones Exteriores de Colombia, desde el 9 de julio de 2025.
- Germán Ávila es el actual Ministro de Hacienda y Crédito Público de Colombia, desde el 20 de julio de 2025.
- Pedro Sánchez es el actual Ministro de Defensa de Colombia, en el cargo desde el 11 de febrero de 2025.
- Edwin Palma es el actual Ministro de Minas y Energía de Colombia, en el cargo desde el 25 de febrero de 2025. 
- Luz Adriana Camargo es la actual fiscal general de Colombia.
- Scott Bessent es el actual Secretario del Tesoro de Estados Unidos.
- Javier Milei es el actual Presidente de Argentina.
- Cristian Quiroz es el Magistrado y Presidente del Consejo Nacional Electoral de Colombia.
- Hernán Penagos Giraldo es el actual Director General de la Registraduría Nacional del Estado Civil de Colombia.
- Gustavo Petro es el actual Presidente de Colombia, en el cargo desde 2022 y hasta 2026.
- El DAPRE es DEPARTAMENTO ADMINISTRATIVO PRESIDENCIA DE LA REPÚBLICA de Colombia. Angie Rodríguez es la directora.
- El 31 de mayo de 2026 se llevará a cabo la primera vuelta de la elección presidencial en Colombia.
- El 21 de julio de 2026 se llevará a cabo la segunda vuelta de la elección presidencial en Colombia.
- El 8 de marzo de 2026 se llevarán a cabo las elecciones legislativas en Colombia, donde se eligirán a los miembros de ambas cámaras del Congreso de Colombia para el periodo 2026-2030.
- El 26 de octubre de 2025 se realizó la consulta presidencial del Pacto Histórico (movimiento político de Gustavo Petro) para escoger el candidato del partido a la presidencia en las elecciones presidenciales de Colombia de 2026. El ganador de la consulta fue el senador Iván Cepeda, obteniendo formalmente el aval para aspirar a la Presidencia de la República.
- El Partido Movimiento de Salvación Nacional respaldó a Abelardo de la Espriella como precandidato, quien a principios de diciembre de 2025 entregó alrededor de 5 millones de firmas ante la Registraduría para inscribir su candidatura a la Presidencia de Colombia.
- El Partido Dignidad y Compromiso será representado por Sergio Fajardo. 
- El Partido Nuevo Liberalismo será representado por Juan Manuel Galán.
- El Partido Verde será representado por Juan Carlos Pinzón, quien también recibió el apoyo del partido político Alianza Democrática Amplia.
- El Partido Fuerza de la Paz será representado por Roy Barreras.
- El Partido En Marcha será representado por Juan Fernando Cristo.
- Camilo Romero es precandidato presidencial y exgobernador del departamento de Nariño.
- La candidata del Partido Centro Democrático (movimiento político de Álvaro Uribe) será definida entre María Fernanda Cabal, Paloma Valencia y Paola Holguín.
- Nicolás Petro es el hijo mayor de Gustavo Petro, quien está en medio de un proceso judicial bajo las acusaciones de lavado de dinero y enriquecimiento ilícito proveniente del narcotráfico el cual tenía como objetivo el financiamiento de la campaña presidencial de su padre.
- La Paz Total es una política de Estado implementada por el gobierno de Gustavo Petro en Colombia, que busca acabar el conflicto armado negociando con diversos grupos armados ilegales abriendo canales con diferentes grupos armados para buscar su desmovilización, acogimiento a la justicia o sometimiento.
- Héctor Alfonso Carvajal, es abogado del presidente Gustavo Petro, y fue nombrado magistrado de la Corte Constitucional de Colombia en mayo de 2025.
- El Clan del Golfo es una organización narcoparamilitar y terrorista que surgió tras la desmovilización entre 2003 y 2006 de las Autodefensas Unidas de Colombia. Tiene presencia en 211 municipios del país y trafica un gran número de cargamentos de droga a nivel nacional e internacional.
- Iván Mordisco es un disidente guerrillero colombiano, parte de las Disidencias de las FARC-EP .
- Diego Marín Buitrago, conocido como “Papá Pitufo”, es un contrabandista que habría intentado infiltrar la campaña de Petro a través de un empresario catalán, Xavier Vendrell, amigo cercano del ahora presidente. La operación habría incluido la entrega de $500 millones de pesos. Petro que rechazó cualquier intento de infiltración. 
- El 30 de noviembre de 2025 se publicó una encuesta de INVAMER que muestra a Iván Cepeda con 31.9% de la intención de voto, Abelardo De La Espriella con 18.2% y Sergio Fajardo con 8.5%. El resto de precandidatos registra menos intención de voto que la opción del voto en blanco.
- En la encuesta de INVAMER del 30 de noviembre, en el escenario de segunda vuelta que enfrente e Cepeda contra De la Espriella, Cepeda saca 59.1%, y De la Espriella 36.2%. Enun esceneario de Cepeda contra Fajardo, Cepeda se queda en 48.9% y Fajardo en 46.4%. En un escenario de Fajardo contra De la Espriella, Fajardo gana con 51.7% mientras que De la Espriella se queda en 38.9%.
- SIC es la Superintendencia de Industria y Comercio.
- Hidroituango es un megaproyecto hidroeléctrico en Antioquia, Colombia, desarrollado por EPM en la que el organismo de control declaró responsables fiscales a 26 personas, incluyendo al Sergio Fajardo, quien fue gobernador cuando se planeó la obra, por los errores que llevaron al colapso de la obra en 2018 debido a obstrucciones generalizadas en los túneles de desviación del río Cauca. La obra terminó pero con retrasos.
- En una serie de videos publicados por el también precandidato presidencial y excontralor de la república Carlos Felipe Córdoba, se evidencia como Sergio Fajardo fue uno de los responsables del desastre de Hidroituango y se relata todo lo sucedido con el proyecto. Fajardo había pedido a la SIC bloquear temporalmente el video publicado por Córdoba porque usa su voz e imagen sin su autorización.
- Recientemente suspendieron al general Juan Miguel Huertas, comandante de Personal del Ejército, por supuestos vínculos entre altos mandos militares y estructuras disidentes de las extintas Fuerzas Armadas Revolucionarias de Colombia (FARC), lideradas por alias “Calarcá”.
- Notica sobre "El Camino de Fajardo", se refiere a un análisis estratégico sobre sus probabilidades de ganar la presidencia, citando la encuesta de INVAMER para decir lo siguiente "La izquierda extrema, la derecha extrema y el centro. Sergio Fajardo llega fuerte a cinco meses de las elecciones y la primera vuelta. El exgobernador de Antioquia es el que le compite mejor a Iván Cepeda en segunda vuelta. Aunque en primera vuelta es el tercer candidato, en la segunda Abelardo pierde con una distancia importante frente a Cepeda". Si ves esa noticia, menciona ese contexto.
- Si ves titulares como "¿De Fajardo a Abelardo?" hacen referencia a columnas de opinión que sintetizan lo siguiente: "Si gana Cepeda se viene el abismo. Por eso, como plantea el expresidente Uribe, necesitamos la unidad del centro a la derecha, de Fajardo a Abelardo."
- A menos de que veas la palabra Hiroituango expresamente en el titular, no la menciones.
"""

def extraer_fechas(pregunta):
    pregunta = pregunta.lower()

    # Caso 1: rango tipo "del 25 al 29 de agosto"
    match = re.search(r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+([a-záéíóú]+)", pregunta)
    if match:
        dia_inicio, dia_fin, mes = match.groups()
        fecha_inicio = dateparser.parse(f"{dia_inicio} {mes}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} {mes}", languages=['es'])
        return fecha_inicio.date(), fecha_fin.date()

    # Caso 2: rango tipo "entre el 25 y el 29 de agosto"
    match = re.search(r"entre\s+el\s+(\d{1,2})\s+y\s+el\s+(\d{1,2})\s+de\s+([a-záéíóú]+)", pregunta)
    if match:
        dia_inicio, dia_fin, mes = match.groups()
        fecha_inicio = dateparser.parse(f"{dia_inicio} {mes}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} {mes}", languages=['es'])
        return fecha_inicio.date(), fecha_fin.date()

    # Caso 3: una sola fecha "el 27 de agosto"
    match = re.search(r"(\d{1,2}\s+de\s+[a-záéíóú]+(?:\s+de\s+\d{4})?)", pregunta)
    if match:
        fecha = dateparser.parse(match.group(), languages=['es'])
        return fecha.date(), fecha.date()

    # Caso 4: sin fecha → None, None
    return None, None

# 2️⃣ Obtener fecha más reciente disponible
def obtener_fecha_mas_reciente(df):
    fecha_max = df["Fecha"].max()
    # Si es pandas Timestamp (tiene método .date), conviértelo
    if hasattr(fecha_max, "date"):
        return fecha_max.date()
    # Si ya es datetime.date, devuélvelo directo
    return fecha_max


# 3️⃣ Detectar sentimiento deseado
def detectar_sentimiento_deseado(pregunta):
    pregunta = pregunta.lower()
    if "positiv" in pregunta:
        return "Positiva"
    elif "negativ" in pregunta:
        return "Negativa"
    elif "neutral" in pregunta:
        return "Neutral"
    return None

# 4️⃣ Extraer entidades (personajes, lugares, categorías)
def extraer_entidades(texto):
    texto_lower = texto.lower()
    personajes_dict = {
        "Cepeda": ["Iván Cepeda"],
        "De la Espriella": ["Abelardo De la Espriella"],
        "Milei": ["presidente de argentina"],
        "Fajardo": ["Sergio Fajardo"],
        "Petro": ["presidente de colombia"],
        "Cabal": ["María Fernanda Cabal"]
    }
    
    encontrados = {"personajes": [],  "categorias": [], "lugares": [],}

    for nombre, sinonimos in personajes_dict.items():
        if any(s in texto_lower for s in [nombre.lower()] + sinonimos):
            encontrados["personajes"].append(nombre)

    return encontrados

# 5️⃣ Filtrar titulares por entidades y sentimiento (versión mejorada)
def filtrar_titulares(df_filtrado, entidades, sentimiento_deseado):
    """
    Filtra titulares usando entidades detectadas y sentimiento, sin depender
    de columnas de cobertura geográfica ni de idioma.

    - Personajes y lugares: se buscan en el título.
    - Categorías: se buscan en la columna 'Término' cuando exista.
    """
    if df_filtrado.empty:
        return pd.DataFrame()

    filtro = df_filtrado.copy()
    condiciones = []

    # Personajes
    if entidades["personajes"]:
        condiciones.append(
            filtro["Título"].str.lower().str.contains(
                "|".join([p.lower() for p in entidades["personajes"]]),
                na=False
            )
        )

    # Lugares
    if entidades["lugares"]:
        condiciones.append(
            filtro["Título"].str.lower().str.contains(
                "|".join([l.lower() for l in entidades["lugares"]]),
                na=False
            )
        )

    # Si hubo condiciones → OR entre todas
    if condiciones:
        filtro = filtro[pd.concat(condiciones, axis=1).any(axis=1)]

    # Filtrar por sentimiento si aplica
    if sentimiento_deseado:
        filtro = filtro[filtro["Sentimiento"] == sentimiento_deseado]

    return filtro

# 7️⃣ Nube de palabras con colores y stopwords personalizadas
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size >= 60:
        return "rgb(255, 205, 0)"
    elif font_size >= 40:
        return "rgb(0, 48, 135)"
    else:
        return "rgb(200, 16, 46)"

def generar_nube(titulos, archivo_salida):
    texto = " ".join(titulos)
    texto = re.sub(r"[\n\r]", " ", texto)
    stopwords = set([
        "dice", "tras", "pide", "va", "día", "Colombia", "elección", "elecciones", "contra", "países",
        "van", "ser", "hoy", "año", "años", "nuevo", "nueva", "será",
        "sobre", "entre", "hasta", "donde", "desde", "como", "pero", "también", "porque", "cuando",
        "ya", "con", "sin", "del", "los", "las", "que", "una", "por", "para", "este", "esta", "estos",
        "estas", "tiene", "tener", "fue", "fueron", "hay", "han", "son", "quien", "quienes", "le",
        "se", "su", "sus", "lo", "al", "el", "en", "y", "a", "de", "un", "es", "si", "quieren", "aún",
        "mantiene", "buscaría", "la", "haciendo", "recurriría", "ante", "meses", "están", "subir",
        "ayer", "prácticamente", "sustancialmente", "busca", "cómo", "qué", "días", "construcción","tariffs",
        "aranceles","construcción", "Sergio","así", "no","Fajardo","irá", "está", "sea", "eso"
    ])
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        color_func=color_func,
        collocations=False,
        max_words=10
    ).generate(texto)
    wc.to_file(archivo_salida)

def seleccionar_titulares_categorizados(noticias_dia, max_total=None):
        """
        Selecciona titulares SOLO de Sergio Fajardo (Término == "Sergio Fajardo"),
        priorizando los más repetidos del día.

        - Si max_total es None: devuelve TODOS (sin límite).
        - Si max_total es número: devuelve hasta max_total.
        """

        if noticias_dia is None or noticias_dia.empty:
            return []

        # 1) Filtrar SOLO Fajardo por columna Término
        if "Término" not in noticias_dia.columns:
            return []

        df_fajardo = noticias_dia[
            noticias_dia["Término"].astype(str).str.strip().str.lower() == "sergio fajardo"
        ].copy()

        if df_fajardo.empty:
            return []

        # 2) Normalizar títulos y calcular repetición
        df_fajardo["titulo_norm"] = (
            df_fajardo["Título"].fillna("").astype(str).str.strip().str.lower()
        )

        # conteo por título (repetidos arriba)
        conteos = df_fajardo["titulo_norm"].value_counts()

        # 3) Para cada título, tomar una fila representativa (la primera) y adjuntar conteo
        filas = []
        for titulo_norm, c in conteos.items():
            fila = df_fajardo[df_fajardo["titulo_norm"] == titulo_norm].iloc[0]
            filas.append({
                "titulo": str(fila.get("Título", "")).strip(),
                "medio": str(fila.get("Fuente", "")).strip(),
                "enlace": fila.get("Enlace", ""),
                "_conteo": int(c),
            })

        # 4) Orden final: más repetidos primero; desempate por medio
        filas.sort(key=lambda x: (-x["_conteo"], x["medio"]))

        # 5) Quitar campo interno y aplicar límite si existe
        seleccion = [{"titulo": f["titulo"], "medio": f["medio"], "enlace": f["enlace"]} for f in filas]

        if isinstance(max_total, int) and max_total > 0:
            return seleccion[:max_total]

        return seleccion

def generar_resumen_y_datos(fecha_str):
    """
    Genera el resumen diario, la nube de palabras y la selección de titulares,
    ahora con estructura temática obligatoria en hasta 3 párrafos, todos orientados a Sergio Fajardo:

    Mantiene:
    - Cache en /resumenes/resumen_{fecha}.txt
    - resumenes_metadata.csv
    - resumenes_index.faiss
    - Subida a S3
    - Nube de palabras
    - Lista de titulares del día
    """
    # Normalizar fecha y filtrar noticias del día
    fecha_dt = pd.to_datetime(fecha_str, errors="coerce").date()
    noticias_dia = df[df["Fecha"].dt.date == fecha_dt]
        # =============================================================================
    # FIRMA DEL DATASET DEL DÍA (para detectar cambios en las noticias)
    # =============================================================================
    import hashlib

    titulos_dia = (
        noticias_dia["Título"]
        .fillna("")
        .str.strip()
        .sort_values()
        .tolist()
    )

    firma_str = "||".join(titulos_dia)
    firma_dataset = hashlib.md5(firma_str.encode("utf-8")).hexdigest()

    if noticias_dia.empty:
        return {"error": f"No hay noticias para la fecha {fecha_str}"}
        # =============================================================================
    # 🔥 PREPARAR TITULARES + NUBE DESDE EL INICIO (para que existan antes de guardar metadata)
    # =============================================================================
    os.makedirs("nubes", exist_ok=True)
    archivo_nube = f"nube_{fecha_str}.png"
    archivo_nube_path = os.path.join("nubes", archivo_nube)

    # Selección de titulares (modo cliente: todos los del día cuyo Término == "Sergio Fajardo")
    titulares_relacionados = []
    if "Término" in noticias_dia.columns:
        df_fajardo_termino = noticias_dia[
            noticias_dia["Término"].astype(str).str.strip().str.lower() == "sergio fajardo"
        ]
    else:
        df_fajardo_termino = noticias_dia.iloc[0:0]

    for _, row in df_fajardo_termino.iterrows():
        titulares_relacionados.append({
            "titulo": row.get("Título", ""),
            "medio": row.get("Fuente", ""),
            "enlace": row.get("Enlace", "")
        })

    titulares_info = seleccionar_titulares_categorizados(noticias_dia, max_total=None)

    # Nube del día (con todos los títulos del día, o si prefieres solo los de fajardo, me dices)
    generar_nube(df_fajardo_termino["Título"].tolist(), archivo_nube_path)


    # =============================================================================
    # 1️⃣ MÁSCARAS TEMÁTICAS
    # =============================================================================
    # Palabras clave para cada bloque
    fajardo_kw = ["sergio fajardo", "fajardo"]
    petro_kw = ["gustavo petro", "petro"]
    cne_kw = ["cne", "consejo nacional electoral", "registraduría"]
    # Otros candidatos (puedes ampliar esta lista a tu gusto)
    otros_candidatos_kw = [
        "Iván Cepeda", "Cepeda","Abelardo de la Espriella","De la Espriella","Miguel Uribe Londoño", 
        "Claudia López","Vicky Dávila","Juan Carlos Pinzón", "Germán Vargas Lleras", "Santiago Botero","Juan Manuel Galán",
        "Aníbal Gaviria","Enrique Peñalosa","María Fernanda Cabal", "Paloma Valencia", "Camilo Romero", "Luis Gilberto Murillo",
        "Luis Carlos Reyes", "Efraín Cepeda","Paola Holguín","Roy Barrera","David Luna", "Mauricio Cárdenas", "Juan Daniel Oviedo",
         "Mauricio Armitage","Carlos Felipe Córdoba","Mauricio Gómez Amín", "Mauricio Lizcano","Daniel Palacio","Juan Fernando Cristo"
    ]
    otros_candidatos_kw = [s.lower() for s in otros_candidatos_kw]

    partidos_kw = [
                    "Elección", "Elecciones","Coalición", "Coaliciones", "Pacto Histórico","Cambio Radical", "Centro Democrático", "Partido de la U", 
                   "Colombia Humana","Alianza Verde","Partido Liberal", "Partido Conservador","Comunes","Nuevo Liberalismo","En Marcha","Dignidad y Compromiso",
                    "Partido MIRA","Ahora Colombia","Movimiento de Salvación Nacional", "Cámara de representantes", "Senado", "Álvaro Uribe", "César Gaviria", "Simón Gaviria", "Manuel Virgüez",
                    "Ana Paola Agudelo","Jorge Robledo", "Antonio Navarro Wolff", "Clara Luz Roldán", "Congreso"
                    ]
    partidos_kw = [s.lower() for s in partidos_kw]

    # Palabras a excluir para "otros candidatos" (institucional/proceso)
    excluir_otros_kw = cne_kw + partidos_kw + fajardo_kw + petro_kw

    def texto_bajo(row):
        titulo = str(row.get("Título", "")).lower()
        termino = str(row.get("Término", "")).lower() if "Término" in row else ""
        return f"{titulo} {termino}"

    def contiene_alguna(texto, palabras):
        return any(p in texto for p in palabras)

    # Construir series booleanas (máscaras)
    textos = noticias_dia.apply(texto_bajo, axis=1)

    mask_fajardo = textos.apply(lambda t: contiene_alguna(t, fajardo_kw))
    mask_petro = textos.apply(lambda t: contiene_alguna(t, petro_kw))
    mask_cne = textos.apply(lambda t: contiene_alguna(t, cne_kw))
    mask_partidos = textos.apply(lambda t: contiene_alguna(t, partidos_kw))

    mask_otros_candidatos = textos.apply(
        lambda t: (
            contiene_alguna(t, otros_candidatos_kw)
            and not contiene_alguna(t, excluir_otros_kw)
        )
    )

    # =============================================================================
    # 2️⃣ CONSTRUCCIÓN DE BLOQUES DE CONTEXTO
    # =============================================================================
    def construir_contexto(mask, label):
        subset = noticias_dia[mask]
        if subset.empty:
            return f"No hubo titulares relevantes sobre {label} en las noticias del día.\n"

        lineas = []
        for _, row in subset.iterrows():
            titulo = str(row.get("Título", "")).strip()
            fuente = str(row.get("Fuente", "")).strip()
            if not titulo:
                continue
            if fuente:
                lineas.append(f"- {titulo} ({fuente})")
            else:
                lineas.append(f"- {titulo}")

        # Limitar a ~10–12 líneas para que el contexto sea manejable
        return "\n".join(lineas)

    contexto_fajardo = construir_contexto(mask_fajardo, "Sergio Fajardo")
    #contexto_petro = construir_contexto(mask_petro, "Gustavo Petro")
    #contexto_cne = construir_contexto(mask_cne, "el CNE / Consejo Nacional Electoral / Registraduría")
    #contexto_otros = construir_contexto(mask_otros_candidatos, "otros candidatos distintos a Fajardo y Petro")
    #contexto_partidos = construir_contexto(mask_partidos, "partidos políticos , coaliciones y congreso")

    # También podemos construir un contexto general con todas las noticias del día
    contexto_todas = "\n".join(
        f"- {row['Título']} ({row['Fuente']})"
        for _, row in noticias_dia.iterrows()
    )

    # =============================================================================
    # 3️⃣ CONTEXTO NARRATIVO PREVIO (sólo días ANTERIORES a la fecha del resumen)
    # =============================================================================
    CONTEXTO_ANTERIOR = ""
    try:
        meta_path = "faiss_index/resumenes_metadata.csv"
        if os.path.exists(meta_path):
            df_prev = pd.read_csv(meta_path)

            if len(df_prev) > 0 and "fecha" in df_prev.columns:
                # Normalizar fechas a tipo date
                df_prev["fecha"] = pd.to_datetime(
                    df_prev["fecha"], errors="coerce"
                ).dt.date

                # Quedarnos SOLO con resúmenes de días anteriores al que vamos a resumir
                df_prev_anteriores = df_prev[df_prev["fecha"] < fecha_dt].sort_values("fecha")

                if len(df_prev_anteriores) > 0:
                    ultimos = df_prev_anteriores.tail(1)
                    contexto_texto = "\n\n".join(
                        f"({row['fecha']}) {str(row['resumen']).strip()}"
                        for _, row in ultimos.iterrows()
                    )

                    CONTEXTO_ANTERIOR = (
                        "CONTEXTO DEL ÚLTIMO DÍA ANTERIOR REGISTRADO:\n"
                        f"{contexto_texto}\n"
                    )

                    print(
                        f"🔗 Contexto narrativo cargado "
                        f"(último día anterior: {ultimos.iloc[-1]['fecha']})"
                    )
    except Exception as e:
        print(f"⚠️ No se pudo cargar el contexto narrativo: {e}")


    # =============================================================================
    # 4️⃣ PROMPT CON 4 PÁRRAFOS OBLIGATORIOS
    # =============================================================================
    prompt = f"""
{CONTEXTO_ANTERIOR}

{CONTEXTO_POLITICO}

Tienes titulares de noticias sobre política colombiana del día {fecha_str}.
Debes redactar un resumen enfocado en todo lo que se diga sobre Sergio Fajardo.
Debes redactar EXACTAMENTE TRES PÁRRAFOS CONTINUOS (sin títulos, sin encabezados, sin numeración):

- En el primer párrafo, menciona la noticia más repetida del día sobre Sergio Fajardo, explicando su contexto y los distintos enfoques de los medios.
- En el segundo párrafo, menciona la segunda noticia más repetida sobre Sergio Fajardo solo si es diferente de la primera. Si no hay una segunda noticia claramente diferenciada (ejemplo, si ves noticias hablando de consultas interpartidistas, deben ir en el mismo párrafo), integra aquí el resto de menciones relevantes del día y termina ahí y utiliza este párrafo para cerrar el panorama general del día.
- En el tercer párrafo, desarrolla la tercera noticia más repetida sobre Sergio Fajardo si es que la hay y es diferente de la primera y segunda. Si no existe una tercera noticia claramente diferenciada, que no haya tercer párrafo y concluye en el segundo.

IMPORTANTE:
- No escribas títulos, encabezados ni etiquetas como “Párrafo 1”, “Párrafo 2” o similares.
- El resultado final debe ser texto corrido, separado únicamente por saltos de línea entre párrafos.

Reglas generales:
- Extensión total hasta 150 palabras. Concreto
- Tono profesional, neutro y orientado a tomadores de decisión.
- El resumen debe referirse EXCLUSIVAMENTE a Sergio Fajardo.
- Si otros actores aparecen en los titulares, solo deben mencionarse en la medida en que afecten directamente a Fajardo.
- No desarrolles secciones ni narrativas independientes sobre otros candidatos, el presidente, partidos o autoridades electorales.
- Prioriza siempre lo ocurrido el {fecha_str}; el contexto de días previos solo sirve para dar continuidad a las narrativas.
- NO inventes hechos ni des tus interpretaciones extrapoles ni deduzcas más allá de lo que sugieren los titulares. Sol

Bloque – Sergio Fajardo:
{contexto_fajardo}



Contexto general del día (todas las noticias):
{contexto_todas}
    """

    # =============================================================================
    # 5️⃣ CACHE DE RESUMEN EN /resumenes
    # =============================================================================
    os.makedirs("resumenes", exist_ok=True)
    archivo_resumen = os.path.join(
        "resumenes",
        f"resumen_{fecha_str}.txt"
    )

    # -----------------------------------------------------------------------------
    # Archivo de firma del resumen (control de cambios del dataset)
    # -----------------------------------------------------------------------------
    archivo_firma = os.path.join(
        "resumenes",
        f"resumen_{fecha_str}_firma.txt"
        )
    # DECISIÓN: ¿REUTILIZAR RESUMEN O REHACER TODO?
    # =============================================================================

    rehacer_resumen = True

    if os.path.exists(archivo_resumen) and os.path.exists(archivo_firma):
        with open(archivo_firma, "r", encoding="utf-8") as f:
            firma_guardada = f.read().strip()

        if firma_guardada == firma_dataset:
            rehacer_resumen = False


    if not rehacer_resumen:
        # -------------------------------------------------------------------------
        # USAR RESUMEN EXISTENTE (dataset no cambió)
        # -------------------------------------------------------------------------
        with open(archivo_resumen, "r", encoding="utf-8") as f:
            resumen_texto = f.read()

    else:
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un analista experto en noticias y política colombiana."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=900
        )

        resumen_texto = respuesta.choices[0].message.content.strip()

        # Guardar resumen nuevo
        with open(archivo_resumen, "w", encoding="utf-8") as f:
            f.write(resumen_texto)

        # Guardar firma del dataset
        with open(archivo_firma, "w", encoding="utf-8") as f:
            f.write(firma_dataset)
        # =============================================================================
        # 9️⃣ EMBEDDINGS ACUMULATIVOS PARA RESÚMENES (FAISS)
        # =============================================================================
        try:
            os.makedirs("faiss_index", exist_ok=True)
            index_path = "faiss_index/resumenes_index.faiss"

            # Generar embedding del resumen del día
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=resumen_texto.strip()
            ).data[0].embedding
            emb_np = np.array([emb], dtype="float32")

            # Si el índice ya existe, cargarlo y agregar nuevo vector
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
                index.add(emb_np)
                print(f"🧩 Embedding agregado al índice existente ({index.ntotal} vectores totales)")
            else:
                dim = len(emb_np[0])
                index = faiss.IndexFlatL2(dim)
                index.add(emb_np)
                print("🆕 Índice FAISS de resúmenes creado")

            faiss.write_index(index, index_path)
            print("💾 Guardado resumenes_index.faiss actualizado")

            r2_upload("resumenes_index.faiss")
            print("☁️ Subido resumenes_index.faiss a S3")

        except Exception as e:
            print(f"⚠️ Error al actualizar embeddings de resúmenes: {e}")
        # =============================================================================
        # 8️⃣ GUARDAR / ACTUALIZAR resumenes_metadata.csv Y SUBIR A S3
        # =============================================================================
        try:
            os.makedirs("faiss_index", exist_ok=True)
            resumen_meta_path = "faiss_index/resumenes_metadata.csv"

            df_resumen = pd.DataFrame([{
                "fecha": str(fecha_dt),
                "archivo_txt": f"resumen_{fecha_str}.txt",
                "nube": archivo_nube,
                "titulares": len(titulares_info),
                "resumen": resumen_texto.strip()
            }])

            # Si ya existe el archivo, lo leemos y agregamos (sin duplicar fechas)
            if os.path.exists(resumen_meta_path):
                df_prev = pd.read_csv(resumen_meta_path)
            else:
                df_prev = pd.DataFrame(columns=["fecha", "archivo_txt", "nube", "titulares", "resumen"])

            if str(fecha_dt) not in df_prev["fecha"].astype(str).values:
                df_total = pd.concat([df_prev, df_resumen], ignore_index=True)
                print(f"🆕 Agregado nuevo resumen para {fecha_dt}")
            else:
                print(f"♻️ Reemplazando resumen existente para {fecha_dt}")
                df_resumen = df_resumen.reindex(columns=df_prev.columns)
                df_prev.loc[df_prev["fecha"].astype(str) == str(fecha_dt), df_prev.columns] = df_resumen.values[0]
                df_total = df_prev

            df_total.to_csv(resumen_meta_path, index=False, encoding="utf-8")
            print(f"💾 Guardado local de resumenes_metadata.csv con {len(df_total)} fila(s) totales")
            r2_upload("resumenes_metadata.csv")
            print("☁️ Subido resumenes_metadata.csv a S3")
        except Exception as e:
            print(f"⚠️ No se pudo guardar/subir resumenes_metadata.csv: {e}")
    # =============================================================================
    # 6️⃣ SELECCIÓN DE TITULARES ALINEADOS A LOS 5 BLOQUES DEL RESUMEN
    # =============================================================================
    def limpiar(texto):
        return re.sub(r"[^a-zA-ZáéíóúñÁÉÍÓÚüÜ0-9 ]", "", str(texto).lower())

    resumen_limpio = limpiar(resumen_texto)
        
    

    
    # MODO CLIENTE: SOLO TITULARES CON Término == "Sergio Fajardo"
    # =============================================================================
    # 6️⃣ TITULARES: MOSTRAR TODOS LOS DEL DÍA CON Término == "Sergio Fajardo"
    # =============================================================================
    #titulares_relacionados = []

    #if "Término" in noticias_dia.columns:
        #df_fajardo_termino = noticias_dia[
            #noticias_dia["Término"].astype(str).str.strip().str.lower() == "sergio fajardo"
        #]
    #else:
        #df_fajardo_termino = noticias_dia.iloc[0:0]  # vacío si no existe la columna

    #for _, row in df_fajardo_termino.iterrows():
        #titulares_relacionados.append({
            #"titulo": row.get("Título", ""),
            #"medio": row.get("Fuente", ""),
            #"enlace": row.get("Enlace", "")
        #})

    # ✅ Sin filtrar por medio, y sin recortar a 12 (el cliente pidió TODOS)



    # =============================================================================
    # 7️⃣ GENERAR NUBE DE PALABRAS
    # =============================================================================
    #os.makedirs("nubes", exist_ok=True)
    #archivo_nube = f"nube_{fecha_str}.png"
    #archivo_nube_path = os.path.join("nubes", archivo_nube)
    #generar_nube(noticias_dia["Título"].tolist(), archivo_nube_path)

    #titulares_info = titulares_relacionados


    # =============================================================================
    # 🔚 RETORNO
    # =============================================================================
    return {
        "resumen": resumen_texto,
        "nube_url": f"/nube/{archivo_nube}",
        "titulares": titulares_info,
    }

@app.route("/resumen", methods=["POST"])
def resumen():
    print("🛰️ Solicitud recibida en /resumen")
    data = request.get_json()
    print(f"📩 JSON recibido: {data}")
    fecha_str = data.get("fecha")
    if not fecha_str:
        return jsonify({"error": "Debe especificar una fecha"}), 400

    resultado = generar_resumen_y_datos(fecha_str)

    if "error" in resultado:
        return jsonify(resultado), 404

    # 🧹 Evitar NaN en la respuesta
    import math
    resultado = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in resultado.items()} if isinstance(resultado, dict) else resultado

    return jsonify(_json_sanitize(resultado))

def extraer_rango_fechas(pregunta):
    # Busca expresiones tipo "entre el 25 y el 29 de agosto"
    match = re.search(r"entre el (\d{1,2}) y el (\d{1,2}) de ([a-zA-Z]+)(?: de (\d{4}))?", pregunta.lower())
    if match:
        dia_inicio, dia_fin, mes, anio = match.groups()
        anio = anio if anio else str(datetime.now().year)
        fecha_inicio = dateparser.parse(f"{dia_inicio} de {mes} de {anio}", languages=['es'])
        fecha_fin = dateparser.parse(f"{dia_fin} de {mes} de {anio}", languages=['es'])
        if fecha_inicio and fecha_fin:
            return fecha_inicio.date(), fecha_fin.date()
    return None, None
MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}
# -----------------------------------------
# 🆕 Helper para obtener semanas reales del mes (lunes–viernes)
# -----------------------------------------
def normalizar_frase_semanas(texto: str) -> str:
    """
    Normaliza frases del tipo:
    - 'entre la primera semana de noviembre y la segunda?'
    - 'entre la primera semana de noviembre y la segunda de noviembre?'

    para que queden como:
    - 'entre la primera semana de noviembre y la segunda semana de noviembre'
    """

    meses_regex = (
        r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|setiembre|octubre|noviembre|diciembre"
    )

    # ¿Hay alguna referencia explícita a 'X semana de <mes>'?
    m = re.search(
        r"(primera|segunda|tercera|cuarta)\s+semana\s+de\s+(" + meses_regex + r")",
        texto,
        re.IGNORECASE,
    )
    if not m:
        return texto  # si no hay semanas del mes, no tocamos nada

    mes = m.group(2)

    # 1) Caso: '... y la segunda de noviembre' -> '... y la segunda semana de noviembre'
    texto = re.sub(
        r"\by\s+la\s+(segunda|tercera|cuarta)\s+de\s+" + mes + r"\b",
        lambda m3: f" y la {m3.group(1)} semana de {mes}",
        texto,
        flags=re.IGNORECASE,
    )

    # 2) Caso: '... y la segunda?' -> '... y la segunda semana de noviembre'
    texto = re.sub(
        r"\by\s+la\s+(segunda|tercera|cuarta)\b(?!\s+semana)",
        lambda m2: f" y la {m2.group(1)} semana de {mes}",
        texto,
        flags=re.IGNORECASE,
    )

    # Limpieza de espacios dobles
    texto = re.sub(r"\s{2,}", " ", texto)

    return texto

def obtener_semanas_del_mes(anio, mes, fecha_min_dataset, fecha_max_dataset):
    """
    Devuelve una lista de rangos semanales reales dentro de un mes:
    - Cada semana inicia en LUNES
    - Cada semana termina en DOMINGO, pero luego se ajusta al dataset
    - Solo se devuelven semanas que tengan algún día dentro del dataset
    """
    semanas = []

    # Primer día del mes
    desde = datetime(anio, mes, 1).date()

    # Último día del mes
    if mes == 12:
        hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
    else:
        hasta = datetime(anio, mes + 1, 1).date() - timedelta(days=1)

    # Mover "desde" al lunes de esa semana
    inicio = desde - timedelta(days=desde.weekday())  # weekday: lunes=0

    while inicio <= hasta:
        fin = inicio + timedelta(days=6)

        # Ajustar al mes
        real_inicio = max(inicio, desde)
        real_fin = min(fin, hasta)

        # Ajustar al dataset
        final_inicio = max(real_inicio, fecha_min_dataset)
        final_fin = min(real_fin, fecha_max_dataset)

        # Si el rango tiene al menos un día válido → agregarlo
        if final_inicio <= final_fin:
            semanas.append((final_inicio, final_fin))

        # Siguiente semana
        inicio += timedelta(days=7)

    return semanas

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}

def interpretar_rango_fechas(pregunta: str, df_noticias: pd.DataFrame):
    """
    Interpreta fechas o rangos mencionados en la pregunta y los ajusta
    al rango disponible en df_noticias.

    Devuelve (fecha_inicio, fecha_fin, origen), donde las fechas son date o None.
    """
    if df_noticias is None or df_noticias.empty:
        return None, None, "sin_datos"

    fechas_validas = df_noticias["Fecha"].dropna()
    if fechas_validas.empty:
        return None, None, "sin_datos"

    fecha_min = fechas_validas.min().date()
    fecha_max = fechas_validas.max().date()

    texto = (pregunta or "")
    texto_lower = texto.lower()
    texto_lower = normalizar_frase_semanas(texto_lower)

    fecha_inicio = None
    fecha_fin = None
    origen = "sin_fecha"

    # 1️⃣ Casos relativos: "esta semana", "hoy", "ayer"
    if fecha_inicio is None and fecha_fin is None:
        if "esta semana" in texto_lower:
            fecha_fin = fecha_max
            fecha_inicio = max(fecha_min, fecha_max - timedelta(days=6))
            origen = "esta_semana_dataset"
        elif re.search(r"\bhoy\b", texto_lower):
            fecha_inicio = fecha_fin = fecha_max
            origen = "hoy_dataset"
        elif re.search(r"\bayer\b(?=[\s,.!?;:]|$)", texto_lower):
            candidata = fecha_max - timedelta(days=1)
            if candidata < fecha_min:
                candidata = fecha_min
            fecha_inicio = fecha_fin = candidata
            origen = "ayer_dataset"

    # 2️⃣ Rango de semanas:
    #    - "entre la primera y la segunda semana de noviembre"
    #    - "entre la primera semana de noviembre y la segunda semana de noviembre"
    #    - "entre la primera semana de noviembre y la segunda de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        # Forma 1: entre la primera y la segunda semana de noviembre
        patron1 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+y\s+"
            r"(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        # Forma 2: entre la primera semana de noviembre y la segunda semana de noviembre
        patron2 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s+y\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+\2",
            texto_lower,
        )
        # Forma 3: entre la primera semana de noviembre y la segunda de noviembre
        patron3 = re.search(
            r"entre\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s+y\s+(?:a\s+la|al|la)\s+(primera|segunda|tercera|cuarta)\s+de\s+\2",
            texto_lower,
        )
        if patron1 or patron2 or patron3:
            if patron1:
                ord1, ord2, nombre_mes = patron1.groups()
            elif patron2:
                # patron2: (ordinal1, mes, ordinal2)
                ord1, nombre_mes, ord2 = patron2.groups()
            else:
                # patron3: (ordinal1, mes, ordinal2)
                ord1, nombre_mes, ord2 = patron3.groups()

            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                anio = fecha_max.year
                # Inicio y fin del mes calendario
                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                # Semanas "fijas" (1–7, 8–14, 15–21, 22–28)
                semanas = [
                    (desde, desde + timedelta(days=6)),                      # primera
                    (desde + timedelta(days=7), desde + timedelta(days=13)), # segunda
                    (desde + timedelta(days=14), desde + timedelta(days=20)),# tercera
                    (desde + timedelta(days=21), desde + timedelta(days=27)) # cuarta
                ]
                ordenes = ["primera", "segunda", "tercera", "cuarta"]
                i1 = ordenes.index(ord1)
                i2 = ordenes.index(ord2)
                idx_min, idx_max = min(i1, i2), max(i1, i2)

                fecha_inicio, _ = semanas[idx_min]
                _, fecha_fin = semanas[idx_max]
                origen = "rango_semanas_mes"



    # 3️⃣ Una sola semana: "primera/segunda/tercera/cuarta semana de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        m_semana_mes = re.search(
            r"(primera|segunda|tercera|cuarta)\s+semana\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower
        )
        if m_semana_mes:
            ord_semana = m_semana_mes.group(1)
            nombre_mes = m_semana_mes.group(2)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                anio = fecha_max.year

                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                semanas = [
                    (desde, desde + timedelta(days=6)),                      # primera
                    (desde + timedelta(days=7), desde + timedelta(days=13)), # segunda
                    (desde + timedelta(days=14), desde + timedelta(days=20)),# tercera
                    (desde + timedelta(days=21), desde + timedelta(days=27)) # cuarta
                ]
                idx = ["primera", "segunda", "tercera", "cuarta"].index(ord_semana)
                fecha_inicio, fecha_fin = semanas[idx]
                origen = "semana_del_mes"

    # 4️⃣ Mes completo: "en noviembre", "durante noviembre"
    if fecha_inicio is None and fecha_fin is None:
        m_mes = re.search(
            r"en\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        if m_mes:
            nombre_mes = m_mes.group(1)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                m_anio = re.search(r"(20\d{2})", texto_lower)
                anio = int(m_anio.group(1)) if m_anio else fecha_max.year

                desde = datetime(anio, mes_num, 1).date()
                if mes_num == 12:
                    hasta = datetime(anio + 1, 1, 1).date() - timedelta(days=1)
                else:
                    hasta = datetime(anio, mes_num + 1, 1).date() - timedelta(days=1)

                fecha_inicio, fecha_fin = desde, hasta
                origen = "mes_completo"

    # 5️⃣ Rangos explícitos "entre el 3 y el 7 de noviembre" / "del 3 al 7 de noviembre"
    if fecha_inicio is None and fecha_fin is None:
        patron_entre = re.search(
            r"entre\s+el\s+(\d{1,2})\s+y\s+el\s+(\d{1,2})\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )
        patron_del = re.search(
            r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)",
            texto_lower,
        )

        m = patron_entre or patron_del
        if m:
            dia1 = int(m.group(1))
            dia2 = int(m.group(2))
            nombre_mes = m.group(3)
            mes_num = MESES_ES.get(nombre_mes)
            if mes_num:
                m_anio = re.search(r"(20\d{2})", texto_lower)
                anio = int(m_anio.group(1)) if m_anio else fecha_max.year

                d_ini = min(dia1, dia2)
                d_fin = max(dia1, dia2)

                try:
                    fecha_inicio = datetime(anio, mes_num, d_ini).date()
                    fecha_fin = datetime(anio, mes_num, d_fin).date()
                    origen = "rango_explicito_texto"
                except ValueError:
                    fecha_inicio = None
                    fecha_fin = None
                    origen = "sin_fecha_valida"

    # 6️⃣ Último intento con search_dates (fecha puntual o rango)
    if fecha_inicio is None and fecha_fin is None and "search_dates" in globals():
        try:
            resultados = search_dates(
                texto,
                languages=["es"],
                settings={"RELATIVE_BASE": datetime.combine(fecha_max, datetime.min.time())},
            ) or []
        except Exception:
            resultados = []

        if resultados:
            # Solo aceptamos fragmentos que tengan algún dígito
            # (evita que expresiones vagas como "últimas encuestas"
            # se tomen como una fecha puntual).
            resultados_filtrados = []
            for frag, fecha_dt in resultados:
                if re.search(r"\d", frag):
                    resultados_filtrados.append((frag, fecha_dt))
                                                
            if resultados_filtrados:
                fechas_detectadas = [r[1].date() for r in resultados_filtrados]
            else:
                fechas_detectadas = []
            if fechas_detectadas:
                if (
                    ("entre " in texto_lower or " del " in texto_lower or "del " in texto_lower or "desde " in texto_lower)
                    and len(fechas_detectadas) >= 2
                ):
                    fecha_inicio = min(fechas_detectadas[0], fechas_detectadas[1])
                    fecha_fin = max(fechas_detectadas[0], fechas_detectadas[1])
                    origen = "rango_explicito_search_dates"
                else:
                    fecha_inicio = fecha_fin = fechas_detectadas[0]
                    origen = "fecha_puntual"


    # 7️⃣ Ajustar al rango del dataset
    if fecha_inicio is not None and fecha_fin is not None:
        original_inicio, original_fin = fecha_inicio, fecha_fin
        fecha_inicio = max(fecha_inicio, fecha_min)
        fecha_fin = min(fecha_fin, fecha_max)

        if fecha_inicio > fecha_fin:
            return None, None, "fuera_rango_dataset"

        if (fecha_inicio, fecha_fin) != (original_inicio, original_fin):
            origen += "_ajustada_dataset"

    return fecha_inicio, fecha_fin, origen



def filtrar_docs_por_rango(docs, fecha_inicio, fecha_fin):
    """
    Filtra una lista de Document de LangChain por metadata['fecha'] en el rango dado.
    Devuelve (docs_filtrados, se_aplico_filtro: bool).
    """
    if not docs or not fecha_inicio or not fecha_fin:
        return docs, False

    filtrados = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        fecha_meta = meta.get("fecha")
        if not fecha_meta:
            continue
        try:
            f = pd.to_datetime(fecha_meta).date()
        except Exception:
            continue
        if fecha_inicio <= f <= fecha_fin:
            filtrados.append(d)

    if filtrados:
        return filtrados, True
    else:
        # Si el filtro deja todo vacío, devolvemos la lista original
        # para no quedarnos sin contexto.
        return docs, False

#pregunta!!!!    
# ------------------------------
# 🤖 Endpoint /pregunta (RAG con filtro por fecha antes de FAISS)
# ------------------------------
@app.route("/pregunta", methods=["POST"])
def pregunta():
    """
    Chatbot principal (versión LangChain).

    - Interpreta fechas/rangos y, si existen noticias en ese periodo,
      construye un mini-vectorstore FAISS SOLO con esas noticias.
    - Si NO hay noticias en ese rango o no se menciona fecha, usa el
      vectorstore global con k=40 (más contexto).
    - Usa retriever_resumenes para contexto macro (resúmenes diarios).
    """
    data = request.get_json()
    q = data.get("pregunta", "").strip()
    if not q:
        return jsonify({"error": "No se proporcionó una pregunta válida."}), 400

    try:
        # 🧠 1️⃣ Detectar entidades y rango de fechas
        entidades = extraer_entidades(q) if "extraer_entidades" in globals() else {}
        fecha_inicio, fecha_fin, origen_rango = interpretar_rango_fechas(q, df)
        print(f"📅 Rango interpretado para la pregunta: {fecha_inicio} → {fecha_fin} ({origen_rango})")

        tiene_rango = fecha_inicio is not None and fecha_fin is not None

        # 🧠 2️⃣ Filtrar DataFrame por rango ANTES de FAISS (solo si hay rango)
        df_rango = pd.DataFrame()
        if tiene_rango:
            # Asegurarnos de trabajar solo con filas que sí tienen fecha
            df_validas = df.dropna(subset=["Fecha"]).copy()
            df_validas["Fecha_date"] = pd.to_datetime(df_validas["Fecha"], errors="coerce").dt.date

            mask = (df_validas["Fecha_date"] >= fecha_inicio) & (df_validas["Fecha_date"] <= fecha_fin)
            df_rango = df_validas[mask].copy()

            print(f"🧾 Noticias en rango {fecha_inicio} → {fecha_fin}: {len(df_rango)} filas")

        # 🧠 3️⃣ Recuperar resúmenes relevantes (contexto macro)
        resumen_docs = []
        if retriever_resumenes is not None:
            try:
                # Compatibilidad con distintas versiones de LangChain:
                if hasattr(retriever_resumenes, "get_relevant_documents"):
                    resumen_docs = retriever_resumenes.get_relevant_documents(q)
                else:
                    resumen_docs = retriever_resumenes.invoke(q)
            except Exception as e:
                print(f"⚠️ Error al recuperar resúmenes con LangChain: {e}")
                resumen_docs = []
        else:
            print("⚠️ retriever_resumenes es None (aún no hay resúmenes indexados).")

        # Filtrar resúmenes por rango (con fallback si deja todo vacío)
        resumen_docs_filtrados, _ = filtrar_docs_por_rango(resumen_docs, fecha_inicio, fecha_fin)

        bloques_resumen = []
        dias_resumen_usados = []
        for d in resumen_docs_filtrados:
            texto = d.page_content.strip()
            if len(texto) > 600:
                texto = texto[:600] + "..."
            fecha_meta = d.metadata.get("fecha") if d.metadata else None
            if fecha_meta:
                dias_resumen_usados.append(fecha_meta)
                bloques_resumen.append(f"[Resumen {fecha_meta}]\n{texto}")
            else:
                bloques_resumen.append(f"[Resumen sin fecha]\n{texto}")

        bloque_resumenes = "\n\n".join(bloques_resumen) if bloques_resumen else "No se encontraron resúmenes relevantes."

        # 🧠 4️⃣ Recuperar noticias relevantes (contexto micro)
        noticias_docs_filtrados = []

        # 4.A) Si hay rango y SÍ hay noticias en el rango → mini-vectorstore temporal
        if tiene_rango and not df_rango.empty:
            print("🧩 Usando mini-vectorstore temporal de noticias dentro del rango solicitado.")

            docs_rango = []
            for _, row in df_rango.iterrows():
                titulo = str(row.get("Título", "")).strip()
                if not titulo:
                    continue

                fecha_val = row.get("Fecha_date") or row.get("Fecha")
                if pd.notnull(fecha_val):
                    try:
                        fecha_str = pd.to_datetime(fecha_val).strftime("%Y-%m-%d")
                    except Exception:
                        fecha_str = None
                else:
                    fecha_str = None

                metadata = {
                    "fecha": fecha_str,
                    "fuente": row.get("Fuente"),
                    "enlace": row.get("Enlace"),
                    "sentimiento": row.get("Sentimiento"),
                    "termino": row.get("Término"),
                }

                docs_rango.append(Document(page_content=titulo, metadata=metadata))

            if docs_rango:
                mini_vs = LCFAISS.from_documents(docs_rango, embeddings)
                mini_ret = mini_vs.as_retriever(search_kwargs={"k": 40})
                try:
                    if hasattr(mini_ret, "get_relevant_documents"):
                        noticias_docs_filtrados = mini_ret.get_relevant_documents(q)
                    else:
                        noticias_docs_filtrados = mini_ret.invoke(q)
                except Exception as e:
                    print(f"⚠️ Error al recuperar noticias con mini-vectorstore: {e}")
                    noticias_docs_filtrados = []
            else:
                print("⚠️ No se construyeron documentos para el mini-vectorstore (rango vacío tras limpieza).")
                noticias_docs_filtrados = []

        # 4.B) Si NO hay rango o NO hay noticias en ese rango → usar vectorstore global (k=40)
        if (not tiene_rango) or (tiene_rango and df_rango.empty):
            if tiene_rango and df_rango.empty:
                print("ℹ️ No hay noticias en el rango pedido; uso vectorstore global como fallback.")
            else:
                print("ℹ️ Pregunta sin fechas claras; uso vectorstore global con k=40.")

            noticias_docs = []
            if 'vectorstore_noticias' in globals() and vectorstore_noticias is not None:
                try:
                    retriever_global = vectorstore_noticias.as_retriever(search_kwargs={"k": 40})
                    if hasattr(retriever_global, "get_relevant_documents"):
                        noticias_docs = retriever_global.get_relevant_documents(q)
                    else:
                        noticias_docs = retriever_global.invoke(q)
                except Exception as e:
                    print(f"⚠️ Error al recuperar noticias con vectorstore global: {e}")
                    noticias_docs = []
            else:
                print("⚠️ vectorstore_noticias es None (no se construyó índice global de noticias).")

            # En este caso NO filtramos por fecha otra vez: o no hay rango, o el rango estaba vacío
            noticias_docs_filtrados = noticias_docs

        # 4.C) Si no hay NADA de contexto (ni resúmenes ni noticias), responde orientando
        if not resumen_docs_filtrados and not noticias_docs_filtrados:
            mensaje = (
                "No encontré noticias claramente relacionadas con tu pregunta en el histórico disponible. "
                "Intenta reformularla, por ejemplo:\n"
                "- Especifica un tema (aranceles, tasas de interés, nearshoring, etc.)\n"
                "- Menciona un país, ciudad o personaje.\n"
                "- Si quieres un periodo, indica las fechas aproximadas."
            )
            return jsonify({
                "respuesta": mensaje,
                "titulares_usados": [],
                "filtros": {
                    "entidades": entidades,
                    "rango": [str(fecha_inicio), str(fecha_fin)] if (fecha_inicio and fecha_fin) else None,
                    "resumenes_usados": [],
                }
            })

        # 🧾 5️⃣ Construir bloque de titulares + lista para el frontend
        lineas_titulares = []
        titulares_usados = []
        vistos = set()

        for d in noticias_docs_filtrados:
            titulo = d.page_content.strip()
            meta = d.metadata or {}
            fuente = (meta.get("fuente") or "Fuente desconocida").strip()
            enlace = (meta.get("enlace") or "").strip()
            fecha_meta = meta.get("fecha") or ""

            clave = (titulo, fuente, enlace, fecha_meta)
            if clave in vistos:
                continue
            vistos.add(clave)

            linea = f"- {titulo} ({fuente}, {fecha_meta})".strip()
            lineas_titulares.append(linea)

            titulares_usados.append({
                "titulo": titulo,
                "medio": fuente,
                "fecha": fecha_meta,
                "enlace": enlace,
            })

        lineas_titulares = lineas_titulares[:6]
        titulares_usados = titulares_usados[:6]

        if lineas_titulares:
            bloque_titulares = "\n".join(lineas_titulares)
        else:
            bloque_titulares = "No se encontraron titulares específicos, solo contexto general de resúmenes."

        # 🧠 6️⃣ Construir texto final para la chain de LangChain
        texto_usuario = f"""{CONTEXTO_POLITICO}

Responde en español, de forma clara, profesional y analítica.
Usa ÚNICAMENTE la información contenida en los resúmenes y titulares listados abajo.
Si hay resúmenes o titulares en inglés, traduce y sintetiza su contenido.

IMPORTANTE:
- Si el bloque de "Titulares relevantes" que verás más abajo contiene al menos una viñeta que empiece con "-", significa que SÍ hay noticias para trabajar.
- Si la pregunta menciona explícitamente a dos actores, países o bloques (por ejemplo, "Trump y México", "Estados Unidos y China"), asegúrate de explicar de forma equilibrada qué ocurre con cada uno y cómo se relacionan entre sí dentro del periodo solicitado.
- En ese caso, está TOTALMENTE PROHIBIDO escribir frases como:
  "no se reportaron noticias específicas", "no hubo noticias", 
  "no se registraron titulares", "no se encontraron noticias sobre X"
  o cualquier variante equivalente.
- Aunque las noticias sean pocas, indirectas o tangenciales respecto a la pregunta,
  debes describir con precisión qué sí se sabe a partir de esos titulares y por qué
  son relevantes (por ejemplo, porque afectan a México de forma indirecta, etc.).

Solo si el bloque de titulares contiene literalmente el texto:
"No se encontraron titulares específicos, solo contexto general de resúmenes."
puedes explicar que no hay noticias puntuales y limitarte al contexto general.

Si el contexto realmente no contiene ningún titular ni resumen relacionado con la pregunta, indícalo explícitamente y no inventes datos.
Si sí hay información parcial, responde de todas formas describiendo lo que se puede afirmar a partir de esos titulares, sin exagerar pero tampoco diciendo que no hay información.
No menciones titulares individuales, es decir, si algún titular menciona una noticia, cuenta la noticia, no el hecho de que hay un titular de algún medio hablando de eso.
Evita frases como “no se dispone de información específica”; en su lugar, explica directamente lo que sí muestran los titulares.
A menos que el contexto esté totalmente vacío, contesta con un mínimo de 150 palabras.

Pregunta del usuario:
{q}

Rango temporal de referencia (si aplica):
{fecha_inicio} → {fecha_fin}

Resúmenes relevantes:
{bloque_resumenes}

Titulares relevantes:
{bloque_titulares}

Respuesta:
"""

        # 🧩 7️⃣ Llamada a LangChain (ChatOpenAI + PromptTemplate)
        texto_respuesta = chain_pregunta.invoke({"texto_usuario": texto_usuario}).strip()

        return jsonify({
            "respuesta": texto_respuesta,
            "titulares_usados": titulares_usados,
            "filtros": {
                "entidades": entidades,
                "rango": [str(fecha_inicio), str(fecha_fin)] if (fecha_inicio and fecha_fin) else None,
                "resumenes_usados": dias_resumen_usados,
            }
        })

    except Exception as e:
        print(f"❌ Error en /pregunta (LangChain): {e}")
        return jsonify({"error": str(e)}), 500




#correoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
@app.route("/enviar_email", methods=["POST"])
def enviar_email():
    data = request.get_json()
    email = data.get("email")
    fecha_str = data.get("fecha")
    fecha_dt = pd.to_datetime(fecha_str).date()

    resultado = generar_resumen_y_datos(fecha_str)
    if "error" in resultado:
        return jsonify({"mensaje": resultado["error"]}), 404

    titulares_info = resultado.get("titulares", [])
    resumen_texto = resultado.get("resumen", "")



    if not resumen_texto:
        archivo_resumen = os.path.join("resumenes", f"resumen_{fecha_str}.txt")
        if os.path.exists(archivo_resumen):
            with open(archivo_resumen, "r", encoding="utf-8") as f:
                resumen_texto = f.read()
        # 🔹 Convertir saltos de línea en HTML para conservar párrafos en el correo
    resumen_html = (resumen_texto or "").replace("\n\n", "<br><br>").replace("\n", "<br>")
    # ☁️ Nube
    archivo_nube = os.path.join("nubes", f"nube_{fecha_str}.png")

    # ---- CONFIGURACIÓN DEL CORREO ----
    remitente = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_PASS")

    destinatario = email

    msg = MIMEMultipart()
    msg["From"] = formataddr(("Monitoreo +", remitente))  # 👈 nombre visible
    msg["To"] = destinatario
    msg["Subject"] = f"Resumen de noticias {fecha_str}"

    # 🧱 Titulares en tabla: máximo 4 por fila (compatible con Gmail/Outlook)
    titulares_cards = []
    for t in titulares_info:
        titulo = (t.get("titulo") or "").strip()
        medio = (t.get("medio") or "").strip()
        enlace = (t.get("enlace") or "").strip()

        card = f"""
        <div style="padding:10px; border:1px solid #ddd; border-radius:12px; background:#fff; height:100%;">
            <a href="{enlace}" style="color:#0B57D0; font-weight:600; text-decoration:none;">
                {titulo}
            </a>
            <br>
            <small style="color:#7D7B78;">• {medio}</small>
        </div>
        """
        titulares_cards.append(card)

    filas_html = []
    for i in range(0, len(titulares_cards), 2):
        fila = titulares_cards[i:i+2]

        # celdas de la fila
        tds = "".join([f'<td style="width:25%; padding:6px; vertical-align:top;">{c}</td>' for c in fila])

        # si faltan celdas para completar 4, rellenar con vacías
        faltan = 2 - len(fila)
        if faltan > 0:
            tds += "".join(['<td style="width:25%; padding:6px;"></td>' for _ in range(faltan)])

        filas_html.append(f"<tr>{tds}</tr>")

    titulares_es_html = f"""
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-bottom:20px; border-collapse:collapse;">
        {''.join(filas_html)}
    </table>
    """

    # 📧 Plantilla HTML con estilo
    cuerpo = f"""
    
    <table role="presentation" cellspacing="0" cellpadding="0" border="0" align="center" style="width:100%; max-width:800px; font-family:Montserrat,Arial,sans-serif; border-collapse:collapse; margin:auto;">
    <!-- Header con fondo blanco -->
    <tr>
        <td style="background:#fff; padding:16px 20px; border-bottom:2px solid #e5e7eb;">
        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
            <tr>
                <td align="right" style="font-weight:700; font-size:1.2rem; color:#111;">
                    Monitoreo<span style="color:#FFB429;">+</span>
                </td>
            </tr>
        </table>
        </td>
    </tr>

    <!-- Bloque gris con contenido -->
    <tr>
        <td style="background:#f9f9f9; padding:20px; border:1px solid #e5e7eb; border-radius:0 0 12px 12px;">
        
        <!-- Resumen -->
        <h2 style="font-size:1.4rem; font-weight:700; margin-bottom:14px; color:#111;">
            📅 Resumen diario de noticias — {fecha_str}
        </h2>
        <div style="background:#fff; border:1px solid #ddd; border-radius:12px; padding:20px; margin-bottom:20px;">
            <p style="color:#555; line-height:1.7; text-align:justify;">{resumen_html}</p>
        </div>

        <!-- Titulares español -->
        <h3 style="font-size:1.15rem; font-weight:700; color:#555; margin-top:20px;">🗞️ Principales titulares</h3>
        {titulares_es_html}
        <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:20px;">
            {''.join([
                f"<div style='padding:10px; border:1px solid #ddd; border-radius:12px; background:#fff; max-width:100%; "
                f"word-break:normal; white-space:normal; overflow-wrap:anywhere;'>"
                f"<a href='{t['enlace']}' style='color:#0B57D0; font-weight:600; text-decoration:none;'>{t['titulo']}</a>"
                f"<br><small style='color:#7D7B78;'>• {t['medio']}</small></div>"
                 for t in titulares_info
            ])}
        </div>

        <!-- Nube -->
        <h3 style="font-size:1.15rem; font-weight:700; color:#555; margin-top:20px;">☁️ Nube de palabras</h3>
        <div style="text-align:center; margin-top:12px;">
            <img src="cid:nube" alt="Nube de palabras" style="width:100%; max-width:600px; border-radius:12px; border:1px solid #ddd;" />
        </div>

        </td>
    </tr>
    </table>
    """


    msg.attach(MIMEText(cuerpo, "html"))
    
    # 📎 Adjuntar nube inline
    if os.path.exists(archivo_nube):
        with open(archivo_nube, "rb") as img_file:
            imagen = MIMEImage(img_file.read())
            imagen.add_header("Content-ID", "<nube>")
            imagen.add_header("Content-Disposition", "inline", filename=archivo_nube)
            msg.attach(imagen)
      
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Gmail
        server.starttls()
        server.login(remitente, password)
        server.sendmail(remitente, destinatario, msg.as_string())  # 👈 enviar
        server.quit()
        return jsonify({"mensaje": f"✅ Correo enviado a {destinatario}"})
    
    except Exception as e:
    
        return jsonify({"mensaje": f"❌ Error al enviar correo: {e}"})



@app.route("/nube/<filename>")
def serve_nube(filename):
    return send_from_directory("nubes", filename)

@app.route("/fechas", methods=["GET"])
def fechas():
    global df
    try:
        if df.empty:
            print("⚠️ DataFrame vacío al solicitar /fechas")
            return jsonify([])

        # Normalizar tipo de dato (maneja tanto datetime64 como date)
        if pd.api.types.is_datetime64_any_dtype(df["Fecha"]):
            fechas_unicas = df["Fecha"].dropna().dt.date.unique()
        else:
            # Si ya son objetos date o strings convertibles
            fechas_unicas = pd.to_datetime(df["Fecha"], errors="coerce").dropna().dt.date.unique()

        fechas_ordenadas = sorted(fechas_unicas, reverse=True)
        fechas_str = [f.strftime("%Y-%m-%d") for f in fechas_ordenadas]

        print(f"🗓️ /fechas → {len(fechas_str)} fechas detectadas (rango {fechas_str[-1]} → {fechas_str[0]})")
        return jsonify(fechas_str)

    except Exception as e:
        print(f"❌ Error en /fechas: {e}")
        return jsonify([])




# ------------------------------
# 📑 Endpoint para análisis semanal
# ------------------------------
@app.route("/reporte_semanal", methods=["GET"]) 
def reporte_semanal():
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reporte_semanal")
    os.makedirs(carpeta, exist_ok=True)

    archivos = [
        f for f in os.listdir(carpeta)
        if f.lower().endswith(".pdf")
    ]
    archivos.sort(reverse=True)  # más recientes primero

    resultados = []
    for f in archivos:
        # Extraer fechas del nombre (ej: analisis_2025-08-25_a_2025-08-29.pdf)
        match = re.search(r"(\d{4}-\d{2}-\d{2})_a_(\d{4}-\d{2}-\d{2})", f)
        if match:
            fecha_inicio = datetime.strptime(match.group(1), "%Y-%m-%d")
            fecha_fin = datetime.strptime(match.group(2), "%Y-%m-%d")
            nombre_bonito = f"Reporte semanal: {fecha_inicio.day}–{fecha_fin.day} {nombre_mes(fecha_fin)}"
        else:
            nombre_bonito = f  # fallback al nombre del archivo

        resultados.append({
            "nombre": nombre_bonito,
            "url": f"/reporte/{f}"
        })

    return jsonify(resultados)

@app.route("/reporte/<path:filename>", methods=["GET"])
def descargar_reporte(filename):
    return send_from_directory("reporte_semanal", filename, as_attachment=False)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
