# Image Classification Model Using Transfer Learning

## Description
This project utilizes the pre-trained VGG16 model for image classification with additional fine-tuning on a custom dataset. The model includes improvements such as dropout, batch normalization, and a reduced learning rate to enhance accuracy.

## Dataset
### Context
This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:

- **Fruits**: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango
- **Vegetables**: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant

### Content
The dataset is organized into three main folders:

- **Train**: Contains 100 images per category.
- **Test**: Contains 10 images per category.
- **Validation**: Contains 10 images per category.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Matplotlib

Install the required dependencies:
```bash
pip install tensorflow matplotlib
