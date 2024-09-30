import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from utils import augment_malignant_images
from sklearn.preprocessing import LabelEncoder

class SkinLesionDataset(Dataset):
    def __init__(self, basic_path=None, train_ratio=0.7, val_ratio=0.1, mode="training", augment=True,
                 num_augmented_per_image=3, transform=None, num_classes=2, benign_malignant_ratio=1):
        self.basic_path = basic_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.mode = mode  # Indicates whether the dataset is being used for training, validation, or testing
        self.transform = transform
        self.augment = augment
        self.num_augmented_per_image = num_augmented_per_image
        self.num_classes = num_classes
        self.benign_malignant_ratio = benign_malignant_ratio

        # Load Metadata
        data_list_path = os.path.join(basic_path, r"isic-2024-challenge\train-metadata.csv")
        data_list = pd.read_csv(data_list_path)

        # Splitting the data into training, validation, and test sets
        malignant_samples = data_list[data_list['target'] == 1]
        benign_samples = data_list[data_list['target'] == 0].sample(malignant_samples.shape[0] * (1 + self.num_augmented_per_image), random_state=42)
        balanced_data = pd.concat([malignant_samples, benign_samples]).reset_index(drop=True)
        X_temp, X_test, y_temp, y_test = train_test_split(
            balanced_data, balanced_data['target'], test_size=self.test_ratio, random_state=42, stratify=balanced_data['target'])
        total_size = len(balanced_data)

        val_updated_ratio = (total_size * self.val_ratio) / len(X_temp)

        # Splitting benign and malignant cases for further processing
        benign_data = X_temp[X_temp['target'] == 0]
        malignant_data = X_temp[X_temp['target'] == 1]
        print(f"Sum of original malignant lesions:{len(malignant_data)}")

        # Augment malignant images if necessary
        augmented_dir = os.path.join(basic_path, 'augmented_malignant')
        augment_malignant_images(self.basic_path, malignant_data, augmented_dir, self.num_augmented_per_image)

        # Count malignant images (original + augmented)
        num_malignant = len(malignant_data) * (1 + self.num_augmented_per_image)
        print(f"Total malignant images after augmentation: {num_malignant}")
        print(f"Sum of original benign lesions:{len(benign_data)}")

        # Load augmented malignant images from the augmented directory
        augmented_images = []
        for idx, row in malignant_data.iterrows():
            for i in range(self.num_augmented_per_image):
                augmented_images.append({
                    'isic_id': f"aug_{row['isic_id']}_{i}",
                    'target': 1  # All augmented images are malignant
                })
        augmented_df = pd.DataFrame(augmented_images)

        # Combine original malignant, benign, and augmented malignant data
        X_temp_updated = pd.concat([benign_data, malignant_data, augmented_df])
        class_counts = X_temp_updated['target'].value_counts()
        print(class_counts)


        X_train, X_val, y_train, y_val = train_test_split(
            X_temp_updated, X_temp_updated['target'], test_size=val_updated_ratio, random_state=42, stratify=X_temp_updated['target'])

        if mode == "training":
            self.case_list = X_train
        elif mode == "validation":
            self.case_list = X_val
        else:  # Test mode
            self.case_list = X_test

        self.data = self.case_list.reset_index()

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        accession = self.case_list.iloc[idx]['isic_id']
        # Determine whether the image is augmented or original
        if 'aug_' in accession:
            # If the image is augmented, it's stored in the augmented directory
            img_path = os.path.join(self.basic_path, r'\augmented_malignant', accession + ".jpg")
        else:
            # If the image is original, it's stored in the original directory
            img_path = os.path.join(self.basic_path, r'\isic-2024-challenge\train-image\image', f'{accession}.jpg')

        # Load the image
        img = np.array(Image.open(img_path).convert('RGB')) / 255

        # Convert label to one-hot vector
        label = np.zeros(self.num_classes)
        label[self.data.loc[idx, 'target']] = 1

        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label)
        return img, label


class SkinLesionDatasetMultiModal(Dataset):
    def __init__(self, basic_path=None, train_ratio=0.7, val_ratio=0.1, mode="training", augment=True,
                 num_augmented_per_image=3, transform=None, num_classes=2, benign_malignant_ratio=1):
        self.basic_path = basic_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.mode = mode  # Indicates whether the dataset is being used for training, validation, or testing
        self.transform = transform
        self.augment = augment
        self.num_augmented_per_image = num_augmented_per_image
        self.num_classes = num_classes
        self.benign_malignant_ratio = benign_malignant_ratio

        # Load Metadata
        data_list_path = os.path.join(basic_path, r"isic-2024-challenge\train-metadata.csv")
        data_list = pd.read_csv(data_list_path)

        # Splitting the data into training, validation, and test sets
        malignant_samples = data_list[data_list['target'] == 1]
        benign_samples = data_list[data_list['target'] == 0].sample(malignant_samples.shape[0] * (1 + self.num_augmented_per_image), random_state=42)
        balanced_data = pd.concat([malignant_samples, benign_samples]).reset_index(drop=True)
        X_temp, X_test, y_temp, y_test = train_test_split(
            balanced_data, balanced_data['target'], test_size=self.test_ratio, random_state=42, stratify=balanced_data['target'])
        total_size = len(balanced_data)

        val_updated_ratio = (total_size * self.val_ratio) / len(X_temp)

        # Splitting benign and malignant cases for further processing
        benign_data = X_temp[X_temp['target'] == 0]
        malignant_data = X_temp[X_temp['target'] == 1]
        print(f"Sum of original malignant lesions:{len(malignant_data)}")

        # Augment malignant images if necessary
        augmented_dir = os.path.join(basic_path, 'augmented_malignant')
        augment_malignant_images(self.basic_path, malignant_data, augmented_dir, self.num_augmented_per_image)

        # Count malignant images (original + augmented)
        num_malignant = len(malignant_data) * (1 + self.num_augmented_per_image)
        print(f"Total malignant images after augmentation: {num_malignant}")
        print(f"Sum of original benign lesions:{len(benign_data)}")

        # Load augmented malignant images from the augmented directory
        augmented_images = []
        for idx, row in malignant_data.iterrows():
            for i in range(self.num_augmented_per_image):
                updated_row = row.copy()
                augmented_images.append(updated_row)
                augmented_images[-1]['isic_id'] = f"aug_{updated_row['isic_id']}_{i}"
        augmented_df = pd.DataFrame(augmented_images)

        # Combine original malignant, benign, and augmented malignant data
        X_temp_updated = pd.concat([benign_data, malignant_data, augmented_df])
        class_counts = X_temp_updated['target'].value_counts()
        print(class_counts)


        X_train, X_val, y_train, y_val = train_test_split(
            X_temp_updated, X_temp_updated['target'], test_size=val_updated_ratio, random_state=42,
            stratify=X_temp_updated['target'])

        if mode == "training":
            self.case_list = X_train
        elif mode == "validation":
            self.case_list = X_val
        else:  # Test mode
            self.case_list = X_test

        self.data = self.case_list.reset_index()

    def preprocess_tabular_data(self):
        updated_tabular_data = self.case_list.copy()
        FEATURES = ['age_approx', 'sex', 'anatom_site_general', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext',
                    'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L',
                    'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA',
                    'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_location',
                    'tbp_lv_location_simple', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border',
                    'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
                    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z']
        CATEGORICAL_FEATURES = ['sex', 'anatom_site_general', 'tbp_lv_location', 'tbp_lv_location_simple']
        updated_tabular_data = updated_tabular_data[FEATURES]
        # handling missing data in categorical columns (sex feature)
        most_frequent = updated_tabular_data['sex'].mode()[0]
        updated_tabular_data['sex'].fillna(most_frequent, inplace=True)

        # Fill missing values for 'age_approx' with class-specific medians
        median_age = round(updated_tabular_data['age_approx'].median())
        updated_tabular_data['age_approx'].fillna(median_age, inplace=True)

        encoder = LabelEncoder()
        encoded_df = updated_tabular_data.copy()
        for column in CATEGORICAL_FEATURES:
            encoded_df[column] = encoder.fit_transform(updated_tabular_data[column])
        return encoded_df

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        accession = self.case_list.iloc[idx]['isic_id']
        # Determine whether the image is augmented or original
        if 'aug_' in accession:
            # If the image is augmented, it's stored in the augmented directory
            img_path = os.path.join(self.basic_path, r'\augmented_malignant', accession + ".jpg")
        else:
            # If the image is original, it's stored in the original directory
            img_path = os.path.join(self.basic_path, r'\isic-2024-challenge\train-image\image', f'{accession}.jpg')

        # Load the image
        img = np.array(Image.open(img_path).convert('RGB')) / 255

        # Convert label to one-hot vector
        label = np.zeros(self.num_classes)
        label[self.data.loc[idx, 'target']] = 1

        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label)

        # Load the tabular features
        encoded_df = self.preprocess_tabular_data()
        tabular_features = encoded_df.iloc[idx]
        tabular_features = torch.tensor(tabular_features.values, dtype=torch.float32)
        return img, tabular_features, label
