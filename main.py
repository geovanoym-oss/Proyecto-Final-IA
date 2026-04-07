# Nombre: Geovan Yassil Perez Encarnacion
# Matricula: 24-EISN-2-037

import torch
from torchvision import transforms
from PIL import Image
# aqui importe la estructura del modelo para poder cargar el checkpoint
from entrenamiento import CNN 

def probar_foto(ruta_imagen):
    checkpoint = torch.load("modelo/modelo.pth")
    clases = checkpoint['clases']
    model = CNN(len(clases))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    img = Image.open(ruta_imagen).convert("RGB")
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_t = t(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)
    
    # La IA "investiga" el uso basado en el nombre detectado
    # En este archivo main, solo imprimimos el nombre de la clase
    print(f"La IA dice que es: {clases[pred.item()]}")