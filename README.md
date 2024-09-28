# Skin Cancer Detection Using 3D Total Body Photography

## Project Overview

This repository contains the code and methodologies for detecting skin cancer using 3D Total Body Photography (3D-TBP) images. The project is inspired by the ISIC 2024 Kaggle Challenge, focusing on identifying malignant lesions from 3D-TBP images. Our approach leverages deep learning models, data augmentation, and metadata integration to improve classification accuracy while minimizing computational demands.
## Key Features
- **Pre-trained Models**: ResNet, GoogLeNet, and MobileNetV2.
- **Handling Class Imbalance**: Uses data augmentation and undersampling techniques.
- **Multimodal Approach**: Combines both image and metadata for enhanced classification.
- **Partial AUC**: Evaluates model performance above an 80% true positive rate.

### Prerequisites

You will need the following libraries and tools installed:
- Python 3.8+
- PyTorch
- torchvision
- pandas
- scikit-learn
- matplotlib
- tqdm

### Folder Structure
You will need to create a folder named `model`, which contains two subfolders: `model/weights` and `model/checkpoints`.

### Dataset
The dataset used in this project is sourced from the [ISIC 2024 Kaggle Challenge](https://www.kaggle.com/competitions/isic-2024-challenge/data).


Download the dataset from Kaggle and extract it into your project directory, for example:
![image](https://github.com/user-attachments/assets/179309f0-c543-467d-94b5-dd887cfddba1)

The dataset contains:
- `train-image/` - image files for the training set (provided for train only)
- `train-image.hdf5` - training image data contained in a single hdf5 file, with the `isic_id` as key
- `train-metadata.csv` - metadata for the training set
- `test-image.hdf5` - test image data contained in a single hdf5 file, with the `isic_id` as key
- `test-metadata.csv` - metadata for the test subset
- `sample_submission.csv` - a sample submission file in the correct format.

**Note:** The dataset is not included in this repository. You must download it from Kaggle and place it in the appropriate directories.

### Configuring Paths
Before running the scripts, make sure to update the following paths in the code:
1. **`basic_path`**: This should be set to the path where the dataset (images and metadata) is located on your machine. Update it in the following scripts:
    - `main.py`
    - `main_test.py`
    - `main_multimodal.py`
    - `main_test_multimodal.py`
      
2. **`model_path`**: This should be set to the path where you want to store model weights and checkpoints. Update it in the following scripts:
    - `main.py`
    - `main_multimodal.py`

### Train the model:
To train the model, run: `main.py`  
This will train the model using the specified pre-trained architecture and save the best model weights to the `model/weights/` directory. The training process will handle class imbalance through data augmentation and undersampling.

### Test the model:
To test the model, run: `main_test.py`  
This will evaluate the model on the test dataset and print metrics like accuracy, F1-score, and partial AUC.

   **Note:** If you want to use the **Multimodal Approach**, you will need to use `main_multimodal.py` and `main_test_multimodal.py`.
   
## Detailed Code Explanation

**utils.py** - This file contains utility functions for data transformation, augmentation, and model evaluation.  
- `ToTensor`: Converts images into PyTorch tensors, adjusting the channel order to (Channels, Height, Width).
- `ImageResize`: Resizes images to a specified height and width.
- `check_augmentation_exists`: Checks if augmented images already exist in the directory.
- `augment_malignant_images`: Applies transformations like flipping and rotation to augment malignant images and saves them in the `augmented_malignant/` directory.
- `calc_confusion_matrix`: Computes the confusion matrix elements (TP, TN, FP, FN).
- `calc_f1_score`: Calculates the F1 score based on TP, FP, and FN.
- `partial_auc`: Computes the partial AUC above an 80% true positive rate.
 
**model.py** - This file defines the architecture for the models used in the project.  
- `ModifiedResNet`: A ResNet50 model with additional depthwise separable convolution layers for improved performance and lower computational costs.
- `ModifiedGoogLeNet`: Similar to ResNet, GoogLeNet is enhanced with additional convolution layers for better accuracy.
- `ModifiedMobileNetV2`: A lightweight model tailored for low-resource environments, further optimized with depthwise separable convolutions.
- `MultiModalModel`: A multimodal model combining image and metadata features for enhanced classification accuracy.
 
**dataset.py** - This file contains the dataset loader that handles loading the images and associated metadata. It supports data augmentation, undersampling for benign lesions, and preprocessing of metadata.  
- `SkinLesionDataset`: Loads and preprocesses images and metadata for training, validation, and testing.
- `augment_malignant_images`: Augments malignant images if the dataset is imbalanced.
- `preprocess_tabular_data`: Processes metadata features and applies label encoding for categorical variables like `sex` and `anatom_site_general`.
 
**main.py** - The main training script that runs the training process for the model. It:  
- Loads the dataset and initializes the model.
- Handles data augmentation and undersampling.
- Trains the model for a specified number of epochs and saves the best weights based on validation accuracy.
 
**main_test.py** - This script evaluates the trained model on the test dataset. It:  
- Loads the test images and metadata.
- Evaluates the model and calculates metrics like accuracy, F1-score, and partial AUC.
 
**main_multimodal.py** - This script extends the training process to handle both image data and metadata. It:  
- Uses the `MultiModalModel` that combines image features and metadata.
- Trains the model using both types of data.
 
**main_test_multimodal.py** - This script tests the multimodal model (image + metadata) on the test dataset.



