# Skin Disease Classification Using Transfer Learning in Convolutional Neural Networks

## Overview
This project focuses on the automated classification of various skin diseases using deep learning techniques, specifically Convolutional Neural Networks (CNNs). By leveraging transfer learning, the system aims to enhance diagnostic support in clinical dermatology, providing accurate and reliable assessments of skin conditions such as eczema, psoriasis, and melanoma.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Installation
To run this project, you need to have Python 3.x installed along with the following libraries:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Gradio
- scikit-learn

You can install the required libraries using pip:
```bash
pip install tensorflow keras numpy matplotlib gradio scikit-learn
```

## Usage
1. **Upload your Kaggle API key**: 
   - Upload the `kaggle.json` file to authenticate and download datasets.
   
2. **Download the Dataset**:
   - The dataset can be downloaded from Kaggle using the Kaggle API.

3. **Run the Jupyter Notebook**:
   - Open the Jupyter Notebook and execute the cells sequentially to preprocess the data, train the model, and evaluate its performance.

4. **Gradio Interface**:
   - The project includes a Gradio interface that allows users to upload skin images and receive predictions on the skin condition.

## Project Structure
```
.
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── efficientnet_model.h5
│   └── resnet_model.h5
├── notebooks/
│   └── skin_disease_classification.ipynb
└── requirements.txt
```

## Model Training
The project utilizes several pre-trained models including:
- ResNet50
- InceptionV3
- EfficientNetV2S

### Training Process
- The models are trained using the `ImageDataGenerator` for data augmentation.
- Class weights are computed to handle class imbalance.
- Early stopping and learning rate reduction callbacks are used to optimize training.

### Example Code Snippet
```python
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator with data augmentation
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.3,
    rotation_range=30,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

# Fit the model
model.fit(train_generator, validation_data=val_generator, epochs=60, class_weight=class_weights)
```

## Results
The models achieved high accuracy in classifying skin diseases, with the ensemble method yielding the best performance. The results include metrics such as accuracy, precision, recall, and F1-score.

### Example Output
```
EfficientNetV2 Test Accuracy: 0.3338, Loss: 2.2931
Classification Report:
              precision    recall  f1-score   support
...
```

## Future Work
- Expand the dataset to include more diverse skin tones and conditions.
- Integrate the model into clinical workflows via APIs.
- Implement longitudinal tracking for monitoring skin conditions over time.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
```
