import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import io

# Load trained model
model = tf.keras.models.load_model('mobilenet_food_classification.h5')

# Define class names
class_names = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
               'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']

# USDA API key and URL
USDA_API_KEY = "gibuT7fRpl3mfMoCR7BD3PxLfTLyVfHcnZBtX3zC"
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

#preprocess the given image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

#predict the food item
def predict_food(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    food_item = class_names[class_idx]
    return food_item

#fetch information from USDA API
def get_nutrition_info(food_item):
    params = {
        "api_key": USDA_API_KEY,
        "query": food_item,
        "pageSize": 1
    }
    response = requests.get(USDA_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'foods' in data and len(data['foods']) > 0:
            return data['foods'][0]
    return None

#st title
st.title("Food Nutrition Chatbot")

#upload Image
uploaded_file = st.file_uploader("Upload an image of the food item", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    #display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    #classify the item
    st.write("Classifying the food item...")
    food_item = predict_food(img)
    st.write(f"Predicted food item: **{food_item}**")
    
    #fetch information
    st.write(f"Fetching nutritional information for **{food_item}**...")
    food_data = get_nutrition_info(food_item)
    
    if food_data:
        st.write("### Nutritional Information")
        st.write(f"**Description**: {food_data.get('description')}")
        if 'foodNutrients' in food_data:
            for nutrient in food_data['foodNutrients']:
                st.write(f"{nutrient['nutrientName']}: {nutrient['value']} {nutrient['unitName']}")
    else:
        st.write(f"Sorry, no nutritional information found for **{food_item}**.")

