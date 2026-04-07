import torch
from torchvision import transforms
from PIL import Image
# importe la estructura del modelo desde el archivo de entrenamiento
from entrenamiento import CerebroIA

def probar_foto(ruta_imagen):
    """
    Carga el modelo entrenado y predice qué hay en la imagen proporcionada.
    """
    try:
        # cargamos el archivo generado en entrenamiento.py
        checkpoint = torch.load("modelo/modelo.pth")
        clases = checkpoint['clases']
        
        # aqui inicio la arquitectura con el numero de clases guardadas
        model = CerebroIA(len(clases))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval() # modo evaluacion

        #aqui preparo la imagen para la IA
        img = Image.open(ruta_imagen).convert("RGB")
        t = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ])
        
        # agrego la dimension de lote (batch) necesaria para PyTorch
        img_t = t(img).unsqueeze(0)

        # aqui realizo la prediccion sin calcular gradientes
        with torch.no_grad():
            output = model(img_t)
            _, pred = torch.max(output, 1)
        
        print(f"La IA dice que es: {clases[pred.item()]}")
        
    except FileNotFoundError:
        print("Error: No se encontró 'modelo.pth' en la carpeta /modelo.")