from dataset import SkinLesionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from model import NeuralNet
import torch.nn as nn
from utils import *
from torchvision import transforms
from xception import Xception, get_xception_based_model
import torchvision


def main_test(basic_path):
    batch_size = 1
    # Create Dataset
    transform = transforms.Compose([ToTensor(), ImageResize()])
    test_dataset = SkinLesionDataset(basic_path, mode='test', transform=transform)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trained_model = torchvision.models.googlenet(pretrained=True)
    for paras in trained_model.parameters():
        paras.requires_grad = False
    trained_model.fc = nn.Sequential(
        nn.Linear(in_features=1024, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=2, bias=True),
        nn.Sigmoid()
    )
    # trained_model = get_xception_based_model()
    trained_model.load_state_dict(torch.load(r'C:\Users\lbsal\deep_learning\model\weights\skin_lesion_model.pt'))

    TP_test = TN_test = FP_test = FN_test = 0
    all_labels = []
    all_outputs = []

    trained_model.eval()
    for images_test, labels_test in tqdm(test_loader):
        images_test = images_test
        labels_test = labels_test
        # Forward pass
        outputs = trained_model(images_test)

        # Calculate test accuracy
        # predicted_test = torch.argmax(outputs.data, 1)
        # total_test += len(labels_test)
        # correct_test += (predicted_test == labels_test).sum().item()
        predicted_test = torch.argmax(outputs.data, 1)
        labels_test = torch.argmax(labels_test.data, 1)
        TP, TN, FP, FN = calc_confusion_matrix(labels_test, predicted_test)
        TP_test += TP
        TN_test += TN
        FP_test += FP
        FN_test += FN

        # Accumulate labels and outputs for AUC calculation
        all_labels.extend(labels_test.cpu().numpy())  # Convert labels to numpy and accumulate
        all_outputs.extend(predicted_test.cpu().numpy())  # Get probability for class 1 and accumulate

    # Calculate accuracy
    test_accuracy = calc_f1_score(TP_test, FP_test, FN_test)
    print(f'test accuracy: {test_accuracy:.4f}')

    # Calculate Partial AUC above 80% TPR
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    p_auc = partial_auc(all_labels, all_outputs, min_tpr=0.8)
    print(f'test pAUC: {p_auc:.4f}')


if __name__ == '__main__':
    basic_path = r'E:'
    main_test(basic_path)