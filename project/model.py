import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet, self).__init__()
        # Load the pretrained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)

        # Remove the last fully connected layer
        self.resnet.fc = nn.Identity()

        # Add depthwise separable convolution layers
        # First depthwise convolution: input channels = 2048
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)  # depthwise
        self.pointwise1 = nn.Conv2d(2048, 512, kernel_size=1)  # pointwise to reduce channels

        # Second depthwise convolution: input channels = 512
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pointwise2 = nn.Conv2d(512, 256, kernel_size=1)

        self.fc = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet(x)  # Pass input through ResNet50 backbone
        # Ensure the output retains the suitable spatial dimensions before convolutions
        if x.dim() == 2:  # If x is flattened, reshape it to 4D
            x = x.view(x.size(0), 2048, 1, 1)

        x = self.conv1(x)  # Depthwise and pointwise convolution 1
        x = self.pointwise1(x)

        x = self.conv2(x)  # Depthwise and pointwise convolution 2
        x = self.pointwise2(x)

        x = self.fc(x)  # Final classification layer
        x = x.squeeze()  # Squeeze unnecessary dimensions

        # Apply sigmoid for binary classification
        x = torch.sigmoid(x)  # Ensures output is a probability between 0 and 1
        return x


class ModifiedGoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedGoogLeNet, self).__init__()
        # Load the pretrained GoogLeNet model
        self.googlenet = models.googlenet(pretrained=True)

        # Remove the last fully connected layer
        self.googlenet.fc = nn.Identity()

        # Add depthwise separable convolution layers
        # First depthwise convolution: input channels = 1024
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # depthwise
        self.pointwise1 = nn.Conv2d(1024, 512, kernel_size=1)  # pointwise to reduce channels

        # Second depthwise convolution: input channels = 512, groups = 512
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pointwise2 = nn.Conv2d(512, 256, kernel_size=1)

        # final layer
        self.fc = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.googlenet(x)  # Pass input through GoogLeNet backbone
        # Ensure the output retains suitable spatial dimensions before convolutions
        if x.dim() == 2:  # If x is flattened, reshape it to 4D
            x = x.view(x.size(0), 1024, 1, 1)

        x = self.conv1(x)  # Depthwise and pointwise convolution 1
        x = self.pointwise1(x)

        x = self.conv2(x)  # Depthwise and pointwise convolution 2
        x = self.pointwise2(x)
        x = self.fc(x)  # Final classification layer

        # Apply sigmoid for binary classification
        x = x.squeeze()  # Squeeze unnecessary dimensions
        x = torch.sigmoid(x)  # Ensures output is a probability between 0 and 1
        return x

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedMobileNetV2, self).__init__()
        # Load the pretrained MobileNetV2 model
        self.mobile_net_v2 = models.mobilenet_v2(pretrained=True)

        # Get the number of channels in the last layer of MobileNetV2
        last_channel = self.mobile_net_v2.last_channel  # This should be 1280 for MobileNetV2
        # Replace the classifier layer of MobileNetV2
        self.mobile_net_v2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, 1024)  # Replace with new fully connected layer
        )
        # Add depthwise separable convolution layers
        # First depthwise convolution: input channels = last_channel
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # depthwise
        self.pointwise1 = nn.Conv2d(1024, 512, kernel_size=1)  # pointwise to reduce channels

        # Second depthwise convolution: input channels = 512, groups = 512
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pointwise2 = nn.Conv2d(512, 256, kernel_size=1)

        # Add final classifier
        self.fc = nn.Conv2d(256, num_classes, kernel_size=1)  # Final classifier layer

    def forward(self, x):
        x = self.mobile_net_v2(x)  # Pass input through MobileNetV2 backbone
        # Ensure the output retains suitable spatial dimensions before convolutions
        if x.dim() == 2:  # If x is flattened, reshape it to 4D
            x = x.view(x.size(0), 1024, 1, 1)
        x = self.conv1(x)  # Depthwise and pointwise convolution 1
        x = self.pointwise1(x)

        x = self.conv2(x)  # Depthwise and pointwise convolution 2
        x = self.pointwise2(x)

        x = self.fc(x)  # Final classification layer

        x = x.squeeze()  # Squeeze unnecessary dimensions
        x = torch.sigmoid(x)  # Ensures output is a probability between 0 and 1 for binary classification
        return x

class AdvancedTabularModel(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedTabularModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)  # Adding dropout for regularization

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return x


class MultiModalModel(nn.Module):
    def __init__(self, feature_input_dim, num_classes):
        super(MultiModalModel, self).__init__()
        # Image model
        self.image_model = ModifiedGoogLeNet(num_classes)

        # Enhanced tabular model
        self.tabular_model = AdvancedTabularModel(feature_input_dim)

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(2+64, 128),  # Adjust dimensions accordingly
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular):
        image_features = self.image_model(image)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)  # Convert to shape (1, feature_size)
        tabular_features = self.tabular_model(tabular)
        if tabular_features.dim() == 1:
            tabular_features = tabular_features.unsqueeze(0)  # Convert to shape (1, feature_size)

        combined_features = torch.cat((image_features, tabular_features), dim=1)
        output = self.fusion_layer(combined_features)
        output = torch.sigmoid(output)
        return output

