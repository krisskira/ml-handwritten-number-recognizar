import io
import os
import numpy as np
from fastapi import UploadFile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model_path = os.path.dirname(os.path.abspath(
    __file__)) + "/../../number_reconizer.keras"

model = load_model(model_path)


async def image_to_number(image: UploadFile) -> int:
    content = await image.read()
    pngImageFile = io.BytesIO(content)
    image = Image.open(pngImageFile)

    imageTomodel = load_and_preprocess_image(image)
    # display_selected_images(imageTomodel)

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
        # Convertir a escala de grises
        image = image.convert('L')
        # Redimensionar a 28x28
        image = image.resize((28, 28))
        # Normalizar y aplanar
        img_array = np.array(image) / 255.0
        return img_array.reshape(-1)
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

    # Predicción deldígito


def predict_digit(_image, model):
    """
    Predice el dígito contenido en una imagen usando un modelo preentrenado.

    Parámetros:
    - image (np.ndarray): Imagen preprocesada.
    - numeros: Modelo preentrenado de Keras.

    Retorna:
    - int: Dígito predicho (0-9).
    """
    # Ajustamos la imagen al formato que requiere el modelo (batch de 1 imagen)
    _image = _image.reshape(1, -1)
    prediction = model.predict(_image)
    # Retornar el índice con la mayor probabilidad (el dígito predicho)
    index = np.argmax(prediction)
    predicted_digit = int(index)
    return predicted_digit


def display_selected_images(image1):
    """
    Muestra dos imágenes preprocesadas lado a lado.

    Parámetros:
    - image1 (np.ndarray): imagen.
    """
    plt.figure(figsize=(8, 4))
    # Mostramos la primera imagen
    plt.subplot(1, 2, 1)
    plt.imshow(image1.reshape(28, 28), cmap='gray')
    plt.title('Imagen')
    plt.axis('off')
    plt.show()
