
import kagglehub
import shutil
import os

# Descargar el dataset
path = kagglehub.dataset_download("leonardopena/top-spotify-songs-from-20102019-by-year")

# path normalmente apunta a un archivo comprimido o CSV; si es ZIP, hay que extraerlo
# Supongamos que es un CSV directamente

# Ruta de destino en el directorio actual con el nombre deseado
dest_path = os.path.join(os.getcwd(), "dataset")

# Mover/renombrar el archivo descargado
shutil.move(path, dest_path)

print("Dataset guardado en:", dest_path)

