# Nombre: Geovan Yassil Perez Encarnacion
# Matricula: 24-EISN-2-037

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
# importe la arquitectura para que la IA pueda cargar los pesos
from entrenamiento import CerebroIA

# BASE DE DATOS DE INVESTIGACION 
# informacion para mostrar detalles sobre lo detectado
INVESTIGACION = {
    "Aspirina": "Uso: Analgésico y protector cardiovascular (Bayer 81mg).",
    "Ibuprofen": "Uso: Antiinflamatorio para dolores fuertes y fiebre.",
    "BitaminaB12": "Uso: Vitamina esencial para el sistema nervioso y energía.",
    "Omega": "Uso: Ácido graso que ayuda a la salud del corazón y el cerebro.",
    "Pregabalina": "Uso: Control del dolor neuropático y nervios.",
    "Dulce": "⚠️ ALERTA: Esto es un dulce. No tiene propiedades médicas."
}

# CARGA DEL MODELO
try:
    # aqui se carga los datos del entrenamiento previo
    checkpoint = torch.load("modelo/modelo.pth")
    clases = checkpoint['clases']
    
    # aqui se creo la instancia de la red y cargamos su "conocimiento"
    model = CerebroIA(len(clases))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval() # modo de prediccion activo
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

def identificar_con_camara(imagen_np):
    """
    Recibe la imagen de la cámara, la procesa con la IA y 
    busca la información en la base de datos local.
    """
    if imagen_np is None: return "Error", "No hay imagen", "0%"
    
    img = Image.fromarray(imagen_np).convert("RGB")
    
    # transformaciones identicas a las del entrenamiento
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_t = t(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img_t)
        # se calcula la probabilidad (confianza) del resultado
        probabilidades = F.softmax(out, dim=1)
        conf, pred = torch.max(probabilidades, 1)

    nombre = clases[pred.item()]
    info = INVESTIGACION.get(nombre, "Información no disponible en la base de datos.")
    
    return f"Detectado: {nombre}", f"Investigación: {info}", f"Confianza: {conf.item():.2%}"

# LANZAMIENTO DE LA INTERFAZ 
interface = gr.Interface(
    fn=identificar_con_camara,
    inputs=gr.Image(label="Enfoque el medicamento"),
    outputs=[
        gr.Textbox(label="Resultado"), 
        gr.Textbox(label="Detalles"), 
        gr.Label(label="Nivel de Certeza")
    ],
    title="Identificador de Medicamentos - Proyecto Final",
    description="Coloque la imagen del producto para identidicarlo y obtener información de su uso."
)

if __name__ == "__main__":
    interface.launch()