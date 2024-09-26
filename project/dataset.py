import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch.nn as nn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from utils import augment_malignant_images


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
        augmented_dir = r'E:\isic-2024-challenge\augmented_malignant'
        augment_malignant_images(self.basic_path, malignant_data, augmented_dir, self.num_augmented_per_image)

        # Count malignant images (original + augmented)
        num_malignant = len(malignant_data) * (1 + self.num_augmented_per_image)
        print(f"Total malignant images after augmentation: {num_malignant}")

        # Undersample benign data to match the number of malignant images multiplied by the ratio
        # self.benign_data = self.benign_data.sample(n=num_malignant * self.benign_malignant_ratio, random_state=42)
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
            # # Splitting benign and malignant cases for further processing
            # self.benign_data = X_train[X_train['target'] == 0]
            # self.malignant_data = X_train[X_train['target'] == 1]
            # print(f"Sum of original malignant lesions:{len(self.malignant_data)}")
            #
            # # Augment malignant images if necessary
            # augmented_dir = r'E:\isic-2024-challenge\augmented_malignant'
            # augment_malignant_images(self.basic_path, self.malignant_data, augmented_dir, self.num_augmented_per_image)
            #
            # # Count malignant images (original + augmented)
            # num_malignant = len(self.malignant_data) * (1 + self.num_augmented_per_image)
            # print(f"Total malignant images after augmentation: {num_malignant}")
            #
            # # Undersample benign data to match the number of malignant images multiplied by the ratio
            # # self.benign_data = self.benign_data.sample(n=num_malignant * self.benign_malignant_ratio, random_state=42)
            # print(f"Sum of original benign lesions:{len(self.benign_data)}")
            #
            # # Load augmented malignant images from the augmented directory
            # augmented_images = []
            # for idx, row in self.malignant_data.iterrows():
            #     for i in range(self.num_augmented_per_image):
            #         augmented_images.append({
            #             'isic_id': f"aug_{row['isic_id']}_{i}",
            #             'target': 1  # All augmented images are malignant
            #         })
            # augmented_df = pd.DataFrame(augmented_images)
            #
            # # Combine original malignant, benign, and augmented malignant data
            # self.case_list = pd.concat([self.benign_data, self.malignant_data, augmented_df])
            # class_counts = self.case_list['target'].value_counts()
            # print(class_counts)
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
            img_path = os.path.join(self.basic_path, r'\isic-2024-challenge\augmented_malignant', accession + ".jpg")
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




# class SkinLesionDataset(Dataset):
#     def __init__(self, basic_path=None, train_ratio=0.7, val_ratio=0.1, mode="training", transform=None):
#         self.basic_path = basic_path
#         self.train_ratio = train_ratio
#         self.val_ratio = val_ratio
#         self.test_ratio = 1 - train_ratio - val_ratio
#         self.mode = mode
#         self.transform = transform
#
#         data_list_path = os.path.join(basic_path, r"isic-2024-challenge\train-metadata.csv")  # read file into list
#         data_list = pd.read_csv(data_list_path)
#
#         # Calculate the sizes of each dataset split
#         malignant_samples = data_list[data_list['target'] == 1]
#         benign_samples = data_list[data_list['target'] == 0].head(malignant_samples.shape[0])
#         balanced_data = pd.concat([malignant_samples, benign_samples]).reset_index(drop=True)
#         X_temp, X_test, y_temp, y_test = train_test_split(
#             balanced_data, balanced_data['target'], test_size=self.test_ratio, random_state=42, stratify=balanced_data['target'])
#         total_size = len(balanced_data)
#
#         val_updated_ratio = (total_size * self.val_ratio) / len(X_temp)
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_temp, y_temp, test_size=val_updated_ratio, random_state=42, stratify=y_temp
#         )
#
#         # X_temp, X_test, y_temp, y_test = train_test_split(
#         #     data_list, data_list['target'], test_size=self.test_ratio, random_state=42, stratify=data_list['target'])
#         # total_size = len(data_list)
#         #
#         # val_updated_ratio = (total_size * self.val_ratio) / len(X_temp)
#         # X_train, X_val, y_train, y_val = train_test_split(
#         #     X_temp, y_temp, test_size=val_updated_ratio, random_state=42, stratify=y_temp
#         # )
#
#         if mode == "training":
#             case_list = X_train
#         elif mode == "validation":
#             case_list = X_val
#         else:
#             case_list = X_test
#
#         self.case_list = list(case_list['isic_id'])
#         self.data = case_list.reset_index()
#         self.num_classes = len(data_list['target'].unique())
#
#     def __len__(self):
#         # return len(self.current_set)
#         return len(self.case_list)
#
#     def __getitem__(self, idx):
#         accession = self.case_list[idx]
#         # parse data from parent folder
#         absolute_path = r'isic-2024-challenge\train-image\image'
#         if not (self.basic_path is None):
#             absolute_path = os.path.join(self.basic_path, absolute_path)
#
#         img_path = os.path.join(absolute_path, f'{accession}.jpg')
#         img = np.array(Image.open(img_path).convert('RGB')) / 255
#
#         # convert label to one-hot vector
#         # label = np.zeros(self.num_classes)
#         # label[self.data.loc[idx, 'target']] = 1
#         label = self.data.loc[idx, 'target']
#
#         # Dataset Preprocessing
#         if self.transform:
#             img = self.transform(img)
#
#         # define input dataset and label
#         label = torch.tensor(label)#.to(torch.float32)
#
#         return img, label