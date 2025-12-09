import os
import boto3

# Cliente S3 usando las variables de entorno de Render
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)

BUCKET = os.getenv("AWS_S3_BUCKET")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")


def s3_upload(filename: str):
    """
    Sube un archivo de faiss_index a S3.

    En el backend lo llamas como:
        r2_upload("resumenes_metadata.csv")
        r2_upload("resumenes_index.faiss")

    Así que aquí asumimos que el archivo local está en faiss_index/<filename>.
    """
    if not BUCKET:
        print("⚠️ AWS_S3_BUCKET no está definido, no se sube nada a S3.")
        return

    local_path = os.path.join(FAISS_DIR, filename)
    if not os.path.exists(local_path):
        print(f"⚠️ Archivo local no encontrado para subir: {local_path}")
        return

    # Puedes guardar con el mismo nombre en la raíz del bucket.
    s3_key = filename  # o f"faiss_index/{filename}" si prefieres un prefijo

    try:
        s3.upload_file(local_path, BUCKET, s3_key)
        print(f"📤 Subido a S3: s3://{BUCKET}/{s3_key}")
    except Exception as e:
        print(f"⚠️ Error al subir {local_path} a S3: {e}")


def s3_download_all():
    """
    Descarga desde S3 los archivos clave al arrancar:

    - resumenes_metadata.csv
    - resumenes_index.faiss
    - noticias_lc/noticias_lc_metadata.csv
    - noticias_lc/index.faiss
    - noticias_lc/index.pkl

    y los deja en faiss_index/<archivo>.
    """
    if not BUCKET:
        print("⚠️ AWS_S3_BUCKET no está definido, no se descarga nada de S3.")
        return

    os.makedirs(FAISS_DIR, exist_ok=True)

    archivos = [
        # Resúmenes globales legacy (si aún los usas)
        "resumenes_metadata.csv",
        "resumenes_index.faiss",

        # Resúmenes LangChain (vectorstore de resúmenes diario)
        "resumenes_lc/resumenes_lc_metadata.csv",
        "resumenes_lc/index.faiss",
        "resumenes_lc/index.pkl",

        # Noticias (ya funcionando)
        "noticias_lc/noticias_lc_metadata.csv",
        "noticias_lc/index.faiss",
        "noticias_lc/index.pkl",
    ]


    for fname in archivos:
        local_path = os.path.join(FAISS_DIR, fname)   # p.ej. faiss_index/noticias_lc/index.faiss
        s3_key = fname                                # p.ej. noticias_lc/index.faiss

        # 👇 ESTA LÍNEA ES LA CLAVE: crear carpeta padre si no existe
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            s3.download_file(BUCKET, s3_key, local_path)
            print(f"📥 Descargado desde S3: s3://{BUCKET}/{s3_key} → {local_path}")
        except Exception as e:
            # No es fatal si no existe al principio (primer deploy)
            print(f"⚠️ No se pudo descargar {s3_key} desde S3: {e}")

