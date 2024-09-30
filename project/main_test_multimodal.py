from dataset import SkinLesionDatasetMultiModal
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from torchvision import transforms
from model import *

import warnings
warnings.filterwarnings('ignore')


def main_test(basic_path, model_path):
    total_test = 0
    correct_test = 0
    batch_size = 1
    feature_input_dim = 37
    # Create Dataset
    transform = transforms.Compose([ToTensor(), ImageResize(), transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_dataset = SkinLesionDatasetMultiModal(basic_path, mode='test', transform=transform)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trained_model = MultiModalModel(feature_input_dim, num_classes=2)
    weights_path = os.path.join(model_path, r'weights\skin_lesion_model.pt')
    trained_model.load_state_dict(torch.load(weights_path))

    TP_test = TN_test = FP_test = FN_test = 0
    all_labels = []
    all_outputs = []

    trained_model.eval()
    for images_test, tabular_data_test, labels_test in tqdm(test_loader):
        images_test = images_test
        tabular_data_test = tabular_data_test
        labels_test = labels_test
        # Forward pass
        outputs = trained_model(images_test, tabular_data_test)
        # Apply sigmoid to convert logits to probabilities for binary classification
        outputs = torch.sigmoid(outputs)

        # Calculate test metrics
        predicted_test = torch.argmax(outputs.data, 1).unsqueeze(0)
        labels_test = torch.argmax(labels_test.data, 1)
        TP, TN, FP, FN = calc_confusion_matrix(labels_test, predicted_test)
        TP_test += TP
        TN_test += TN
        FP_test += FP
        FN_test += FN

        # Accumulate labels and outputs for AUC calculation
        all_labels.extend(labels_test.cpu().numpy())  # Convert labels to numpy and accumulate
        all_outputs.extend(outputs.squeeze(0)[1].unsqueeze(0).cpu().detach().numpy())  # Get probability for class 1
        total_test += len(labels_test)
        correct_test += (predicted_test == labels_test).sum().item()

    # Calculate metrics
    f1_score = calc_f1_score(TP_test, FP_test, FN_test)
    test_accuracy = correct_test/total_test
    print(f'test accuracy: {test_accuracy:.4f}, f1 score: {f1_score:.4f}')

    # Calculate Partial AUC above 80% TPR
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    p_auc = partial_auc(all_labels, all_outputs, min_tpr=0.8)
    print(f'test pAUC: {p_auc:.4f}')

    precision = TP_test / (TP_test + FP_test)
    recall = TP_test / (TP_test + FN_test)  # sensitivity
    specificity = TN_test / (TN_test + FP_test)
    print(f'precision: {precision:.4f}, recall: {recall:.4f}, specificity: {specificity:.4f}')


if __name__ == '__main__':
    # change the paths
    basic_path = r'E:'
    model_path = r'C:\Users\lbsal\deep_learning\model'
    main_test(basic_path, model_path)
