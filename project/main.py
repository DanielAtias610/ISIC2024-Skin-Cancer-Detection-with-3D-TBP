from dataset import SkinLesionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from model import NeuralNet
from xception import Xception, get_xception_based_model
import torch.nn as nn
from utils import *
from torchvision import transforms
import torchvision


def main(basic_path, model_path):
    # Hyper Parameters
    input_size = 3 * 256 * 256  # 3 channels (RGB) and the other are spatial dimensions
    num_classes = 2  # 0: benign, 1: malignant
    num_epochs = 20
    batch_size = 20
    learning_rate = 1e-4
    epoch = 0

    # Create Dataset
    transform = transforms.Compose([ToTensor(), ImageResize()])
    train_dataset = SkinLesionDataset(basic_path, transform=transform, augment=True, num_augmented_per_image=3,
                                      num_classes=2, benign_malignant_ratio=1)
    val_dataset = SkinLesionDataset(basic_path, mode='validation', transform=transform, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device")

    #model = NeuralNet(input_size, num_classes=num_classes, feature_size=(20, 10))
    # Model
    model = torchvision.models.googlenet(pretrained=True)
    for paras in model.parameters():
        paras.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(in_features=1024, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=2, bias=True),
        nn.Sigmoid()
    )
    model.to(device)
    # model = get_xception_based_model()

    model.train()

    # Define loss and optimizer
    # Weights for classes [0, 1]
    # class_weights = torch.tensor([0.5, 1.0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    prev_val_accuracy = 0
    accuracy_train_all_epochs = []
    accuracy_val_all_epochs = []
    pauc_train_all_epochs = []
    pauc_val_all_epochs = []
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        total_train = correct_train = 0
        TP_train = TN_train = FP_train = FN_train = 0
        epoch_ground_truth = []
        epoch_predicted = []
        epoch_val_ground_truth = []
        epoch_val_predicted = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            pbar.set_description(f"epoch: {epoch}, batch loss {loss.item():.3f}")

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate train accuracy
            predicted = torch.argmax(outputs.data, 1)
            labels = torch.argmax(labels.data, 1)
            TP, TN, FP, FN = calc_confusion_matrix(labels, predicted)
            TP_train += TP
            TN_train += TN
            FP_train += FP
            FN_train += FN
            # total_train += len(labels)
            # correct_train += (predicted == labels).sum().item()
            epoch_predicted = np.append(epoch_predicted, predicted.detach().numpy())
            epoch_ground_truth = np.append(epoch_ground_truth, labels.detach().numpy())

        # calculate train accuracy
        # train_accuracy = (correct_train * 100) / total_train
        train_accuracy = calc_f1_score(TP_train, FP_train, FN_train)
        accuracy_train_all_epochs += [train_accuracy]
        train_auc = partial_auc(epoch_ground_truth, epoch_predicted)
        pauc_train_all_epochs += [train_auc]

        # validation loop
        model.eval()
        with torch.no_grad():
            total_val = correct_val = 0
            TP_val = TN_val = FP_val = FN_val = 0

            for images_val, labels_val in val_loader:
                images_val = images_val.to(device)
                labels_val = labels_val.to(device)
                # Forward pass
                outputs = model(images_val)
                loss = criterion(outputs, labels_val)

                # Calculate validation accuracy
                predicted_val = torch.argmax(outputs.data, 1)
                labels_val = torch.argmax(labels_val.data, 1)
                TP, TN, FP, FN = calc_confusion_matrix(labels_val, predicted_val)
                TP_val += TP
                TN_val += TN
                FP_val += FP
                FN_val += FN
                # total_val += len(labels_val)
                # correct_val += (predicted_val == labels_val).sum().item()
                epoch_val_predicted = np.append(epoch_val_predicted, predicted_val.detach().numpy())
                epoch_val_ground_truth = np.append(epoch_val_ground_truth, labels_val.detach().numpy())

            # val_accuracy = (correct_val * 100) / total_val
            val_accuracy = calc_f1_score(TP_val, FP_val, FN_val)
            accuracy_val_all_epochs += [val_accuracy]
            val_auc = partial_auc(epoch_val_ground_truth, epoch_val_predicted)
            pauc_val_all_epochs += [val_auc]

            # calculate average accuracy for the epoch
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

    # plot train and validation dice score in each epoch
    # Accuracy score should be calculated on the entire data and not the average on batch_accuracy
    # (the epoch loss can be calculated as an average of all batch_loss)
    plt.figure()
    plt.title("Training and Validation Accuracy")
    # plt.plot(range(num_epochs), accuracy_val_all_epochs, label="Validation", marker='o')
    # plt.plot(range(num_epochs), accuracy_train_all_epochs, label="Train", marker='o')
    plt.plot(range(num_epochs), accuracy_val_all_epochs, label="Validation", marker='o')
    plt.plot(range(num_epochs), accuracy_train_all_epochs, label="Train", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.show()
    plt.savefig('training accuracy.png')

    plt.figure()
    plt.title("Training and Validation pAUC")
    # plt.plot(range(num_epochs), accuracy_val_all_epochs, label="Validation", marker='o')
    # plt.plot(range(num_epochs), accuracy_train_all_epochs, label="Train", marker='o')
    plt.plot(range(num_epochs), pauc_val_all_epochs, label="Validation", marker='o')
    plt.plot(range(num_epochs), pauc_train_all_epochs, label="Train", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("pAUC")
    plt.legend()
    #plt.show()
    plt.savefig('training pAUC.png')
    a=1


if __name__ == '__main__':
    basic_path = r'E:'
    model_path = r'C:\Users\lbsal\deep_learning\model'
    main(basic_path, model_path)
