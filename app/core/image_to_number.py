import io
import os
import numpy as np
from fastapi import UploadFile
from PIL import Image
import numpy as np
from keras.models import load_model

model_path = os.path.dirname(os.path.abspath(
    __file__)) + "/../../number_reconizer.keras"

model = load_model(model_path)

async def image_to_number(image: UploadFile) -> int:
    """
    Convierte una imagen a un número.

    Parámetros:
    - image (UploadFile): Imagen a convertir.

    Retorna:
    - int: Número convertido.
    """
    content = await image.read()
    pngImageFile = io.BytesIO(content)
    image = Image.open(pngImageFile)
    imageTomodel = load_and_preprocess_image(image)
    digit = predict_digit(imageTomodel, model)
    return digit


def load_and_preprocess_image(image):
    """
    Preprocesa una imagen cargada (objeto PIL).
    - Convierte la imagen a escala de grises.
    - Redimensiona a 28x28 píxeles.
    - Normaliza los valores entre 0 y 1.
    - La aplana para que el modelo pueda procesarla.

    Parámetros:
    - image (PIL.Image.Image): Imagen cargada.

    Retorna:
    - np.ndarray: Imagen preprocesada lista para el modelo.
    """
    try:
        image = image.convert('L')
        image = image.resize((28, 28))
        img_array = np.array(image) / 255.0
        return img_array.reshape(1, 28, 28, 1)
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

def predict_digit(_image, model):
    """
    Predice el dígito contenido en una imagen usando un modelo preentrenado.

    Parámetros:
    - image (np.ndarray): Imagen preprocesada.
    - numeros: Modelo preentrenado de Keras.

    Retorna:
    - int: Dígito predicho (0-9).
    """
    prediction = model.predict(_image)
    index = np.argmax(prediction)
    predicted_digit = int(index)
    return predicted_digit
