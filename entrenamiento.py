# Nombre: Geovan Yassil Perez Encarnacion
# Matricula: 24-EISN-2-037

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import re
from PIL import Image

# CONFIGURACION DE LAS RUTAS
BASE_PATH = "dataset"
MODEL_SAVE_PATH = "modelo/modelo.pth"

if not os.path.exists("modelo"): 
    os.makedirs("modelo")

# preparamos las imagenes: las redimensionamos a 224x224 (estándar para visión artificial)
# y las convertimos en Tensores (el formato que entiende PyTorch) [cite: 1]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DatasetMedico(Dataset):
    """
    Esta clase se encarga de entrar a las carpetas de 'medicamentos' y 'dulces',
    leer cada imagen y asignarle una etiqueta basada en su nombre de archivo. 
    """
    def __init__(self, base_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        # escaneamos las subcarpetas principales del proyecto [cite: 2]
        for subfolder in ["medicamentos", "dulces"]:
            folder_path = os.path.join(base_dir, subfolder)
            if not os.path.exists(folder_path): 
                continue
            
            for f in os.listdir(folder_path):
                # solo aceptamos formatos de imagen validos 
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # extraemos el nombre (ej: de 'Aspirina1.jpg' sacamos 'Aspirina') 
                    nombre_clase = re.sub(r'\d+\.?\d*', '', f.split('.')[0]).strip().capitalize()
                    self.samples.append((os.path.join(folder_path, f), nombre_clase))
        
        # creamos una lista unica de clases detectadas (ej: ['Aspirina', 'Dulce', 'Ibuprofen']) [cite: 4]
        self.clases = sorted(list(set([s[1] for s in self.samples])))
        self.clase_to_idx = {nombre: i for i, nombre in enumerate(self.clases)}
        
        print(f"✅ Dataset cargado correctamente.")
        print(f"✅ Clases detectadas en las carpetas: {self.clases}")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        # metodo para obtener una imagen especifica y su etiqueta numerica
        path, nombre = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, self.clase_to_idx[nombre]

# PRUEBA DE CARGA
if __name__ == "__main__":
    # instanciamos el dataset para verificar que todo este en orden
    dataset = DatasetMedico(BASE_PATH, transform=transform)
    print(f"🚀 Total de imágenes listas para procesar: {len(dataset)}")