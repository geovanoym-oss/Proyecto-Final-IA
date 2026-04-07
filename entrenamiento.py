# Nombre: Geovan Yassil Perez Encarnacion
# Matricula: 24-EISN-2-037

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import re

# CONFIGURACION DE LAS RUTAS
BASE_PATH = "dataset"
MODEL_SAVE_PATH = "modelo/modelo.pth"
if not os.path.exists("modelo"): os.makedirs("modelo")

# aqui añado rotacion y brillo para que la IA  no solo memorice
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class DatasetMedico(Dataset):
    def __init__(self, base_dir, transform=None):
        self.samples = []
        self.transform = transform
        for subfolder in ["medicamentos", "dulces"]:
            folder_path = os.path.join(base_dir, subfolder)
            if not os.path.exists(folder_path): continue
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    nombre_clase = re.sub(r'\d+\.?\d*', '', f.split('.')[0]).strip().capitalize()
                    self.samples.append((os.path.join(folder_path, f), nombre_clase))
        
        self.clases = sorted(list(set([s[1] for s in self.samples])))
        self.clase_to_idx = {nombre: i for i, nombre in enumerate(self.clases)}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, nombre = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.clase_to_idx[nombre]

# ARQUITECTURA DE LA IA 
class CerebroIA(nn.Module):
    """
    Definimos las capas de la Red Neuronal Convolucional.
    Extrae patrones visuales (bordes, formas) para identificar el producto. [cite: 5]
    """
    def __init__(self, num_clases):
        super(CerebroIA, self).__init__()
        # capas de extraccion de caracteristicas
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        # capas de clasificacion final
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Evita que el modelo se sobreajuste
            nn.Linear(128, num_clases)
        )
        
    def forward(self, x): 
        return self.classifier(self.features(x))

# PROCESO DE ENTRENAMIENTO
if __name__ == "__main__":
    dataset = DatasetMedico(BASE_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # aqui inicia el modelo con el numero de clases detectadas 
    model = CerebroIA(len(dataset.clases))
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()

    print(f"Iniciando entrenamiento con {len(dataset)} imágenes...")
    
    # ciclo de aprendizaje (70 epocas para asegurar precision) 
    for epoch in range(70):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    
    # guardamos el resultado final para que main.py y app.py puedan usarlo
    torch.save({'state_dict': model.state_dict(), 'clases': dataset.clases}, MODEL_SAVE_PATH)
    print("¡Modelo entrenado y guardado con éxito en la carpeta /modelo!")