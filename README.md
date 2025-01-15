# image-recognition-chatbot
# Food Nutrition Chatbot

This repository contains the code and files necessary for building a food nutrition chatbot using a deep learning model. The chatbot classifies food items from images and fetches their nutritional information from the USDA API.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Requirements](#requirements)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [Who Can Benefit](#who-can-benefit)
- [Use Cases](#use-cases)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project aims to create a web-based application that can:

1. Classify food items from uploaded images using a pre-trained MobileNetV2 model.
2. Fetch nutritional information about the classified food items from the USDA FoodData Central API.

## Features

- Image upload and preprocessing
- Food classification using a deep learning model
- Fetching nutritional information from USDA API
- Web interface built using Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/food-nutrition-chatbot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd food-nutrition-chatbot
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload an image of a food item.
3. The app will classify the food item and display its predicted name.
4. Nutritional information for the food item will be fetched from the USDA API and displayed on the interface.

## Files

- **app.py**: Contains the Streamlit application code for image upload, classification, and fetching nutritional information.
- **model.py**: Contains the code for building, training, and saving the MobileNetV2 model.
- **preprocessing.py**: Contains the data preprocessing pipeline using `ImageDataGenerator` for training and validation data.
- **test.py**: Contains code for evaluating the trained model on a test dataset.
- **requirements.txt**: Lists all the dependencies required for the project.
- **mobilenet\_food\_classification.h5**: The trained MobileNetV2 model file.

## Requirements

- Python 3.x
- TensorFlow
- Streamlit
- NumPy
- Pandas
- Seaborn
- Matplotlib

Install these dependencies using:

```bash
pip install -r requirements.txt
```

## Model Training

1. The model is built using MobileNetV2, which is pre-trained on ImageNet.
2. Additional layers are added for custom classification into 11 food categories.
3. The model is trained on the `food11` dataset with data augmentation.
4. The best model is saved as `mobilenet_food_classification.h5`.

## Dataset

The `food11` dataset is used for training and evaluating the deep learning model. This dataset includes images of various food items, categorized into 11 different classes. The dataset is divided into training, validation, and test sets to ensure robust model training and evaluation.

The different classes are - apple pie, cheese cake, chicken curry, french fries, fried rice, hamburger, hot dog, ice cream, omelette,  pizza, sushi

### Dataset Details

- **Training Set**: Contains a large number of labeled food images for model training.
- **Validation Set**: Used for hyperparameter tuning and model validation during training.
- **Test Set**: A separate set of images used to evaluate the model's performance after training.
- **Data Augmentation**: Techniques like rotation, flipping, and zooming are applied to the training images to improve model generalization.

The dataset covers a wide range of common food items, enabling the model to generalize well across different types of foods.

## Who Can Benefit

This project can be beneficial for:

- **Dietitians and Nutritionists**: Quickly obtain nutritional information for food items to assist in meal planning and dietary recommendations.
- **Health-Conscious Individuals**: Track nutritional intake by identifying food items and understanding their nutritional value.
- **Developers and Data Scientists**: Learn about image classification, deep learning, and API integration in a practical project.
- **Educational Institutions**: Use this project as a teaching tool for AI and machine learning concepts.

## Use Cases

- **Personal Health Applications**: Integrate the chatbot into fitness and health apps to provide users with nutritional information on the go.
- **Restaurants and Food Services**: Offer detailed nutritional information about menu items by simply uploading an image.
- **Research and Development**: Use the project as a foundation for developing more advanced food recognition and nutrition analysis systems.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [USDA FoodData Central API](https://fdc.nal.usda.gov/api-key-signup.html)

