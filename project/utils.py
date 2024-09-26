import torchvision.transforms as tf
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
from PIL import Image, ImageFilter  # For image manipulations and filtering

class ToTensor:
    """
    This class is a data transformation utility that converts a given image or array into a PyTorch tensor.

    * Converts the input data (assumed to be in the format of a NumPy array or similar) into a PyTorch tensor.
    * Reorders the tensor dimensions from (Height, Width, Channels) to (Channels, Height, Width) to comply with
        PyTorch's tensor format conventions.
    * Converts the tensor data type to float32 for consistency in numerical operations during model training and
        evaluation.
    """
    def __init__(self):
        pass

    def __call__(self, data):
        return torch.tensor(data).permute(2, 0, 1).to(torch.float32)  # verify the channel will be the first dimension


class ImageResize:
    """
    This class is a data transformation utility that resizes images to a specified height and width.
    """
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        return tf.Resize((self.size_h, self.size_w))(data)


def check_augmentation_exists(augmented_dir):
    """
    Check if augmented images already exist.
    Returns True if the directory exists and contains files, otherwise False.
    """
    if os.path.exists(augmented_dir) and len(os.listdir(augmented_dir)) > 0:
        print(f"Augmented images found in {augmented_dir}. Skipping augmentation.")
        return True
    else:
        print(f"No augmented images found. Performing augmentation.")
        return False

def augment_malignant_images(basic_path, malignant_data, augmented_dir, num_augmented_per_image=3):
    """
    Augment malignant images if the augmented directory does not already exist.
    Save augmented images to the specified directory.
    """
    if check_augmentation_exists(augmented_dir):
        return

    os.makedirs(augmented_dir, exist_ok=True)

    # Data augmentation transformations (e.g., flipping, rotation, color jitter)
    data_augmentation = tf.Compose([
        tf.RandomHorizontalFlip(p=0.5),
        tf.RandomVerticalFlip(p=0.5),
        tf.RandomRotation(degrees=15),
        tf.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)), # slight translation
        # transforms.Lambda(lambda img: elastic_transform(img)),  # Slight elastic deformation
        # transforms.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        # RandomGammaCorrection(),  # Mild gamma correction
        # MildBlur(),  # Mild blur to add texture variability
        # tf.Resize(size=(256, 256)),
        # tf.ToTensor()
    ])

    for idx, row in malignant_data.iterrows():
        img_path = os.path.join(basic_path, r"\isic-2024-challenge\train-image\image", row['isic_id'] + ".jpg")
        image = Image.open(img_path).convert('RGB')
        # print(f"Original image path: {img_path}")  # ADD THIS LINE TO CHECK ORIGINAL IMAGE LOADING

        # Apply augmentation
        for i in range(num_augmented_per_image):
            augmented_image = data_augmentation(image)
            augmented_image = tf.ToPILImage()(augmented_image)
            # Define the path to save the augmented image in the specified directory

            save_path = os.path.join(augmented_dir, f"aug_{row['isic_id']}_{i}.jpg")
            augmented_image.save(os.path.join(augmented_dir, f"aug_{row['isic_id']}_{i}.jpg"))
            augmented_image.save(save_path)
            # Verify each augmented image is saved properly
            # print(f"Saved augmented image {i+1} for {row['isic_id']} at {save_path}")

    print(f"Augmentation completed. Images saved in {augmented_dir}.")


def calc_confusion_matrix(labels, predicted):
    TP = torch.sum((labels == 1) & (predicted == 1)).item()
    TN = torch.sum((labels == 0) & (predicted == 0)).item()
    FP = torch.sum((labels == 0) & (predicted == 1)).item()
    FN = torch.sum((labels == 1) & (predicted == 0)).item()
    return TP, TN, FP, FN


def calc_f1_score(TP, FP, FN):
    f1 = (2 * TP) / ((2 * TP) + FP + FN)
    return f1


# def partial_auc(y_true, y_scores, min_tpr=0.8):
#     # my implementation
#     # Calculate ROC curve
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#
#     # Find the indices where TPR is above the specified threshold
#     indices = np.where(tpr >= min_tpr)[0]
#
#     if len(indices) == 0:
#         raise ValueError(f"No TPR values above {min_tpr * 100}% found.")
#
#     # Calculate the partial AUC using the trapezoidal rule
#     total_auc = auc(fpr, tpr)
#     filtered_tpr = tpr.copy()
#     filtered_tpr[filtered_tpr >= min_tpr] = min_tpr
#     filtered_auc = auc(fpr, filtered_tpr)
#     p_auc = total_auc - filtered_auc
#
#     # from matplotlib import pyplot as plt
#     # plt.figure()
#     # plt.plot(fpr, tpr)
#     # plt.plot(fpr, updated_tpr)
#     # plt.show()
#     return p_auc

def partial_auc(ground_truth, prediction, min_tpr=0.8):
    # Check if all values in solution['target'] are 1 or 1
    if ground_truth.sum() == len(ground_truth) or ground_truth.sum() == 0.0:
        raise ValueError("")

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(ground_truth - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * prediction

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(fpr, tpr)
    # plt.show()

    return partial_auc