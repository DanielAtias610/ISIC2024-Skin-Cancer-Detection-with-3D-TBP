from dataset import SkinLesionDatasetMultiModal
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import *
from torchvision import transforms
from model import *
import warnings
warnings.filterwarnings('ignore')

def main(basic_path, model_path):
    # Hyper Parameters
    input_size = 3 * 256 * 256  # 3 channels (RGB) and the other are spatial dimensions
    num_classes = 2  # 0: benign, 1: malignant
    num_epochs = 20
    batch_size = 20
    learning_rate = 1e-4
    feature_input_dim = 37

    # Create Dataset
    transform = transforms.Compose([ToTensor(), ImageResize(), transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = SkinLesionDatasetMultiModal(basic_path, transform=transform, augment=True, num_augmented_per_image=3,
                                      num_classes=2, benign_malignant_ratio=1)
    val_dataset = SkinLesionDatasetMultiModal(basic_path, mode='validation', transform=transform, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device")

    model = MultiModalModel(feature_input_dim, num_classes=num_classes)
    model.to(device)

    model.train()

    # Define loss and optimizer
    # Weights for classes [0, 1] respectively
    class_weights = torch.tensor([0.5, 1])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    prev_val_accuracy = 0
    accuracy_train_all_epochs = []
    accuracy_val_all_epochs = []
    pauc_train_all_epochs = []
    pauc_val_all_epochs = []
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        TP_train = TN_train = FP_train = FN_train = 0
        epoch_ground_truth = []
        epoch_predicted = []
        epoch_val_ground_truth = []
        epoch_val_predicted = []
        for i, (images, tabular_data, labels) in enumerate(train_loader):
            images = images.to(device)
            tabular_data = tabular_data.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images, tabular_data)
            loss = criterion(outputs, labels)
            pbar.set_description(f"epoch: {epoch}, batch loss {loss.item():.3f}")

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate train metrics
            predicted = torch.argmax(outputs.data, 1)
            labels = torch.argmax(labels.data, 1)
            TP, TN, FP, FN = calc_confusion_matrix(labels, predicted)
            TP_train += TP
            TN_train += TN
            FP_train += FP
            FN_train += FN
            epoch_predicted = np.append(epoch_predicted, outputs.data[:, 1].unsqueeze(0).detach().numpy())
            epoch_ground_truth = np.append(epoch_ground_truth, labels.detach().numpy())

        # calculate train f1-score and pAUC
        train_accuracy = calc_f1_score(TP_train, FP_train, FN_train)
        accuracy_train_all_epochs += [train_accuracy]
        train_auc = partial_auc(epoch_ground_truth, epoch_predicted)
        pauc_train_all_epochs += [train_auc]

        # validation loop
        model.eval()
        with torch.no_grad():
            TP_val = TN_val = FP_val = FN_val = 0

            for images_val, tabular_data_val, labels_val in val_loader:
                images_val = images_val.to(device)
                tabular_data_val = tabular_data_val.to(device)
                labels_val = labels_val.to(device)
                # Forward pass
                outputs = model(images_val, tabular_data_val)
                loss = criterion(outputs, labels_val)

                # Calculate validation accuracy
                predicted_val = torch.argmax(outputs.data, 1)
                labels_val = torch.argmax(labels_val.data, 1)
                TP, TN, FP, FN = calc_confusion_matrix(labels_val, predicted_val)
                TP_val += TP
                TN_val += TN
                FP_val += FP
                FN_val += FN
                epoch_val_predicted = np.append(epoch_val_predicted, outputs.data[:, 1].unsqueeze(0).detach().numpy())
                epoch_val_ground_truth = np.append(epoch_val_ground_truth, labels_val.detach().numpy())

            val_accuracy = calc_f1_score(TP_val, FP_val, FN_val)
            accuracy_val_all_epochs += [val_accuracy]
            val_auc = partial_auc(epoch_val_ground_truth, epoch_val_predicted)
            pauc_val_all_epochs += [val_auc]

            # if accuracy improved save model parameter ... state dict...
            if val_accuracy > prev_val_accuracy:
                torch.save(model.state_dict(), os.path.join(model_path, r'weights\skin_lesion_model.pt'))

                # save checkpoints
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, os.path.join(model_path, r'checkpoints\skin_lesion_model.pt'))
                prev_val_accuracy = val_accuracy
        model.train()
        print(f'epoch: {epoch}, train accuracy: {train_accuracy:.4f}, validation accuracy: {val_accuracy:.4f}, pAUC train: {train_auc:.4f}, , pAUC val: {val_auc:.4f}')
        epoch += 1

    # plot train and validation f1-score  and pAUC in each epoch
    plt.figure()
    plt.title("Training and Validation F1-Score")
    plt.plot(range(num_epochs), accuracy_val_all_epochs, label="Validation", marker='o')
    plt.plot(range(num_epochs), accuracy_train_all_epochs, label="Train", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.savefig('training f1-score.png')

    plt.figure()
    plt.title("Training and Validation pAUC")
    plt.plot(range(num_epochs), pauc_val_all_epochs, label="Validation", marker='o')
    plt.plot(range(num_epochs), pauc_train_all_epochs, label="Train", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("pAUC")
    plt.legend()
    plt.savefig('training pAUC.png')


if __name__ == '__main__':
    # change the paths
    basic_path = r'E:'
    model_path = r'C:\Users\lbsal\deep_learning\model'
    main(basic_path, model_path)
