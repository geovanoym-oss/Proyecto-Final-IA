# Nombre: Geovan Yassil Perez Encarnacion
# Matricula: 24-EISN-2-037

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# base de datos de informacion medica para el usuario
INVESTIGACION = {
    "Aspirina": "Uso: Analgésico y protector cardiovascular (Bayer 81mg).",
    "Ibuprofen": "Uso: Antiinflamatorio para dolores fuertes y fiebre.",
    "BitaminaB12": "Uso: Vitamina esencial para el sistema nervioso y energía.",
    "Omega3": "Uso: Ácido graso que ayuda a la salud del corazón y el cerebro.",
    "Pregabalina": "Uso: Control del dolor neuropático y nervios.",
    "Dulce": "⚠️ ALERTA: Esto es un dulce. No tiene propiedades médicas."
}
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

# carga de datos
checkpoint = torch.load("modelo/modelo.pth")
clases = checkpoint['clases']
model = CerebroIA(len(clases))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def identificar_con_camara(imagen_np):
    img = Image.fromarray(imagen_np).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = t(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img_t)
        conf, pred = torch.max(F.softmax(out, dim=1), 1)

    nombre = clases[pred.item()]
    info = INVESTIGACION.get(nombre, "Información no disponible en la base de datos.")
    
    return f"Detectado: {nombre}", f"Investigación: {info}", f"Confianza: {conf.item():.2%}"

gr.Interface(
    fn=identificar_con_camara,
    inputs=gr.Image(label="Enfoque el medicamento"),
    outputs=[gr.Textbox(label="Resultado"), gr.Textbox(label="Detalles"), gr.Label(label="Nivel de Certeza")],
    title="Identificador de Medicamentos - Geovan Perez"
).launch()