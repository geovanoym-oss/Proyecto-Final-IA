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

# configuracion de carpetas
BASE_PATH = "dataset"
MODEL_SAVE_PATH = "modelo/modelo.pth"
if not os.path.exists("modelo"): os.makedirs("modelo")

# transformaciones para mejorar el aprendizaje 
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
        
        # aqui se escanea las carpetas "medicamentos" y "dulces"
        for subfolder in ["medicamentos", "dulces"]:
            folder_path = os.path.join(base_dir, subfolder)
            if not os.path.exists(folder_path): continue
            
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # sacamos el nombre base (ej: "Aspirina" de "Aspirina1.jpg")
                    nombre_clase = re.sub(r'\d+\.?\d*', '', f.split('.')[0]).strip().capitalize()
                    self.samples.append((os.path.join(folder_path, f), nombre_clase))
        
        # aqui creo lista de clases unicas
        self.clases = sorted(list(set([s[1] for s in self.samples])))
        self.clase_to_idx = {nombre: i for i, nombre in enumerate(self.clases)}
        print(f"Clases detectadas: {self.clases}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, nombre = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.clase_to_idx[nombre]

# aqui esta la arquitectura de la red neuronal
class CerebroIA(nn.Module):
    def __init__(self, num_clases):
        super(CerebroIA, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_clases)
        )
    def forward(self, x): return self.classifier(self.features(x))

if __name__ == "__main__":
    dataset = DatasetMedico(BASE_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = CerebroIA(len(dataset.clases))
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()

    print(f"Entrenando con {len(dataset)} imágenes...")
    for epoch in range(70):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    
    torch.save({'state_dict': model.state_dict(), 'clases': dataset.clases}, MODEL_SAVE_PATH)
    print("¡Entrenamiento completado y modelo guardado!")
