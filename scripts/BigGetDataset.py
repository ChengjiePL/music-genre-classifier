
import kagglehub
import shutil
import os

# Descargar el dataset
path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")

# path normalmente apunta a un archivo comprimido o CSV; si es ZIP, hay que extraerlo
# Supongamos que es un CSV directamente

# Ruta de destino en el directorio actual con el nombre deseado
dest_path = os.path.join(os.getcwd(), "../dataset")

# Mover/renombrar el archivo descargado
shutil.move(path, dest_path)

print("Dataset guardado en:", dest_path)

