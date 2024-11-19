import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import json

# Загрузка модели и class_indices
model = load_model('fine_tuned_model.h5')

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Обратное отображение class_indices
class_labels = {v: k for k, v in class_indices.items()}

# Словарь с калорийностью
calorie_dict = {
    'apple': 52,
    'banana': 96,
    'beetroot': 43,
    'bell pepper': 20,
    'cabbage': 25,
    'capsicum': 20,
    'carrot': 41,
    'cauliflower': 25,
    'chilli pepper': 40,
    'corn': 86,
    'cucumber': 16,
    'eggplant': 25,
    'garlic': 149,
    'ginger': 80,
    'grapes': 69,
    'jalepeno': 29,
    'kiwi': 61,
    'lemon': 29,
    'lettuce': 15,
    'mango': 60,
    'onion': 40,
    'orange': 47,
    'paprika': 282,
    'pear': 57,
    'peas': 81,
    'pineapple': 50,
    'pomegranate': 83,
    'potato': 77,
    'raddish': 16,
    'soy beans': 446,
    'spinach': 23,
    'sweetcorn': 86,
    'sweetpotato': 86,
    'tomato': 18,
    'turnip': 28,
    'watermelon': 30
}

# Функция предсказания
def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_name = class_labels[class_idx]
    return class_name, calorie_dict[class_name]

# Интерфейс Streamlit
st.title("Image Classification with VGG16")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    class_name, calories = predict_image(image)
    st.write(f"Predicted Class: {class_name}")
    st.write(f"Calories per 100g: {calories} kcal")
