
# Skin Disease Classification Using Transfer Learning in Convolutional Neural Networks

## Overview
This project focuses on the automated classification of various skin diseases using deep learning techniques, specifically Convolutional Neural Networks (CNNs). By leveraging transfer learning, the system aims to enhance diagnostic support in clinical dermatology, providing accurate and reliable assessments of skin conditions such as eczema, psoriasis, and melanoma.

The project utilizes pre-trained models like ResNet50, InceptionV3, and EfficientNetV2S, fine-tuned on a comprehensive dataset of skin conditions. The solution includes a user-friendly Gradio interface for image uploads and predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
  - [Data Preprocessing](#data-preprocessing)
  - [Transfer Learning](#transfer-learning)
  - [Training Process](#training-process)
- [Results](#results)
  - [Model Performance](#model-performance)
  - [Classification Metrics](#classification-metrics)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation
To run this project, you need to have Python 3.8 or higher installed along with the following libraries:

```bash
pip install tensorflow==2.10.0 keras numpy matplotlib gradio scikit-learn
```

### Additional Requirements
- Kaggle API for dataset download
- Google Colab or Jupyter Notebook for model training
- Gradio for the web interface

## Usage
1. **Upload your Kaggle API key**:
   - Create a `kaggle.json` file and upload it to authenticate and download datasets.

2. **Download the Dataset**:
   - The dataset can be downloaded from Kaggle using the Kaggle API.

3. **Run the Jupyter Notebook**:
   - Open the Jupyter Notebook and execute the cells sequentially to preprocess the data, train the model, and evaluate its performance.

4. **Gradio Interface**:
   - The project includes a Gradio interface that allows users to upload skin images and receive predictions on the skin condition.

### Example Usage
```python
# Example prediction using the Gradio interface
def predict(img):
    img = img.resize((299, 299))  # Resize image for EfficientNet input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_effnet(img_array)  # Preprocess image

    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = CLASS_NAMES[pred_index]
    confidence = preds[0][pred_index]

    return f"{pred_class} ({confidence:.2%} confidence)"
```

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
├── requirements.txt
└── LICENSE
```

## Model Training
The project utilizes several pre-trained models including ResNet50, InceptionV3, and EfficientNetV2S. The models are fine-tuned on a dataset of skin diseases to adapt to the specific classification task.

### Data Preprocessing
- **Data Augmentation**: Techniques like rotation, flipping, zooming, and shearing are applied to increase dataset diversity.
- **Normalization**: Images are normalized using the preprocessing functions from the respective pre-trained models.

### Transfer Learning
- **Base Models**: Pre-trained models are used as feature extractors.
- **Custom Layers**: Additional layers are added on top of the base models to adapt to the skin disease classification task.

### Training Process
- **ImageDataGenerator**: Used for data augmentation and batch processing.
- **Class Weights**: Computed to handle class imbalance.
- **Early Stopping**: Implemented to prevent overfitting.
- **Learning Rate Reduction**: Applied to optimize training.

### Code Snippet
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

### Model Performance
- **EfficientNetV2S**: Test Accuracy: 0.3338, Loss: 2.2931
- **ResNet50**: Test Accuracy: 0.4029, Loss: 2.1242

### Classification Metrics
- **Precision**: Measures the proportion of true positives among all positive predictions.
- **Recall**: Measures the proportion of true positives among all actual positive instances.
- **F1-Score**: Harmonic mean of precision and recall.

### Output
```

Model Performance
  EfficientNetV2S: 
    Test Accuracy: 91%
    Loss: 2.2931
```

### Classification Metrics
The following classification metrics were obtained for the EfficientNetV2S model:
```
| Class                                      | Precision | Recall | F1-Score | Support |
|--------------------------------------------|-----------|--------|----------|--------|
| Acne and Rosacea Photos                    | 0.42      | 0.64   | 0.50     | 312    |
| Actinic Keratosis Basal Cell Carcinoma    | 0.45      | 0.31   | 0.36     | 288    |
| Atopic Dermatitis Photos                   | 0.27      | 0.40   | 0.32     | 123    |
| Bullous Disease Photos                      | 0.17      | 0.14   | 0.16     | 113    |
| Cellulitis Impetigo and other Infections   | 0.12      | 0.16   | 0.14     | 73     |
| Eczema Photos                              | 0.46      | 0.31   | 0.37     | 309    |
| Exanthems and Drug Eruptions              | 0.21      | 0.47   | 0.28     | 101    |
| Hair Loss Photos                           | 0.22      | 0.77   | 0.35     | 60     |
| Herpes HPV and other STDs Photos          | 0.33      | 0.34   | 0.34     | 102    |
| Light Diseases and Disorders of Pigmentation| 0.24      | 0.29   | 0.26     | 143    |
| Lupus and other Connective Tissue diseases  | 0.14      | 0.20   | 0.17     | 105    |
| Melanoma Skin Cancer                       | 0.48      | 0.40   | 0.44     | 116    |
| Nail Fungus and other Nail Disease         | 0.70      | 0.76   | 0.73     | 261    |
| Poison Ivy and other Contact Dermatitis    | 0.23      | 0.14   | 0.17     | 65     |
| Psoriasis and related diseases             | 0.36      | 0.08   | 0.13     | 352    |
| Scabies and other Infestations             | 0.10      | 0.21   | 0.14     | 108    |
| Seborrheic Keratoses and other Benign Tumors| 0.52    | 0.26   | 0.35     | 343    |
| Systemic Disease                           | 0.38      | 0.12   | 0.18     | 152    |
| Tinea and other Fungal Infections          | 0.46      | 0.23   | 0.31     | 325    |
| Urticaria Hives                            | 0.19      | 0.47   | 0.27     | 53     |
| Vascular Tumors                            | 0.18      | 0.31   | 0.23     | 121    |
| Vasculitis Photos                          | 0.24      | 0.47   | 0.32     | 105    |
| Warts and other Viral Infections           | 0.36      | 0.31   | 0.34     | 272    |

Overall Accuracy: 91%
```

### 



## Future Work
- **Expand Dataset**: Include more diverse skin tones and conditions.
- **Clinical Integration**: Develop APIs for integration into clinical workflows.
- **Longitudinal Tracking**: Implement tracking to monitor skin conditions over time.
- **Model Optimization**: Explore quantization and pruning for deployment on edge devices.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- **Kaggle Dataset**: The dataset used in this project is from [Kaggle]().
- **Pre-trained Models**: The project leverages pre-trained models from TensorFlow Applications.
- **Gradio**: The web interface is built using Gradio for easy interaction.

