import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, in_size, num_classes, n_layers=2, feature_size=(200, 100)):
        super(NeuralNet, self).__init__()
        activation = nn.ReLU()
        self.flatten = nn.Flatten()
        layer_list = list()
        layer_list.append(nn.Linear(in_features=in_size, out_features=feature_size[0]))
        layer_list.append(activation)

        for i in range(n_layers - 1):
            layer_list.append(nn.Linear(in_features=feature_size[i], out_features=feature_size[i + 1]))
            layer_list.append(activation)

        layer_list.append(nn.Linear(in_features=feature_size[-1], out_features=num_classes))
        layer_list.append(nn.Softmax())

        self.net = nn.Sequential(*layer_list)
        self.parameter_count()

    def forward(self, x):
        return self.net(self.flatten(x))

    def parameter_count(self):
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f"Number of parameters: {(total_params/1000):.2f}K")
