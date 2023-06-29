
import torch.nn as nn
import torch

class MultiviewCNN(nn.Module):
    def __init__(self, img_width, img_height, coronal_width, input_channel: int, cnn_channels: list, kernel_sizes: list, paddings: list, pool_sizes: list, fc_dims: list, out_dim, regularization):
        """
        Create a normal CNN models. The length of cnn_channels, kernel_sizes, paddings and pool_sizes must be same.
        The input impage is concatenated coronal and sagittal slhs, coronal on the left and sagittal on the right.
        """
        super(MultiviewCNN, self).__init__()

        self.coronal_width = coronal_width
        # Define the convolutional layers
        cnn_channels = [input_channel] + cnn_channels
        self.coronal_convs = nn.ModuleList()
        self.sagittal_convs = nn.ModuleList()

        for i in range(len(cnn_channels)-1):
            coronal_conv = nn.Conv2d(cnn_channels[i], cnn_channels[i+1],kernel_sizes[i], padding=paddings[i])
            coronal_bn = nn.BatchNorm2d(cnn_channels[i+1]) if regularization == "bn" else nn.Identity()
            coronal_relu = nn.ReLU()
            coronal_pool = nn.MaxPool2d(pool_sizes[i])
            self.coronal_convs.extend([coronal_conv, coronal_bn, coronal_relu, coronal_pool])

            sagittal_conv = nn.Conv2d(cnn_channels[i], cnn_channels[i+1],kernel_sizes[i], padding=paddings[i])
            sagittal_bn = nn.BatchNorm2d(cnn_channels[i+1]) if regularization == "bn" else nn.Identity()
            sagittal_relu = nn.ReLU()
            sagittal_pool = nn.MaxPool2d(pool_sizes[i])
            self.sagittal_convs.extend([sagittal_conv, sagittal_bn, sagittal_relu, sagittal_pool])

        # Calculate the final output dimensions after the convolutional layers
        test_coronal = torch.zeros((1, 1, img_height, coronal_width)) # batch_size=1, 1 channel, image width, image height
        test_sagittal = torch.zeros((1, 1, img_height, img_width - coronal_width)) # batch_size=1, 1 channel, image width, image height

        with torch.no_grad():
            for coronal_layer, sagittal_layer in zip(self.coronal_convs, self.sagittal_convs):
                test_coronal = coronal_layer(test_coronal)
                test_sagittal = sagittal_layer(test_sagittal)

            self.coronal_output_size = int(torch.prod(torch.tensor(test_coronal.shape)))
            self.sagittal_output_size = int(torch.prod(torch.tensor(test_sagittal.shape)))

        conv_output_size = self.coronal_output_size + self.sagittal_output_size

        # Define the fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_dims = [conv_output_size] + fc_dims
        for i in range(len(fc_dims) - 1):
            fc_layer = nn.Linear(fc_dims[i], fc_dims[i+1])
            relu_layer = nn.ReLU()
            dropout_layer = nn.Dropout(p=0.25) if regularization == "dropout" else nn.Identity()
            self.fc_layers.extend([fc_layer, relu_layer, dropout_layer])

        self.output_layer = nn.Linear(fc_dims[-1], out_dim)

    def forward(self, x):
        coronal_x = x[:, :, :, 0:self.coronal_width]
        sagittal_x = x[:, :, :, self.coronal_width:]

        # Apply the convolutional layers
        for layer in self.coronal_convs:
            coronal_x = layer(coronal_x)

        for layer in self.sagittal_convs:
            sagittal_x = layer(sagittal_x)
        
        # Flatten the output from the convolutional layers
        coronal_x = coronal_x.view(-1, self.coronal_output_size)
        sagittal_x = sagittal_x.view(-1, self.sagittal_output_size)

        x = torch.cat((coronal_x, sagittal_x), dim=1)
        
        # Apply the fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        x = self.output_layer(x)
        
        return x


class CNN(nn.Module):
    def __init__(self, img_width, img_height, input_channel: int, cnn_channels: list, kernel_sizes: list, paddings: list, pool_sizes: list, fc_dims: list, out_dim, regularization):
        """
        Create a normal CNN models. The length of cnn_channels, kernel_sizes, paddings and pool_sizes must be same.
        """
        super(CNN, self).__init__()
        
        # Define the convolutional layers
        cnn_channels = [input_channel] + cnn_channels
        self.conv_layers = nn.ModuleList()
        for i in range(len(cnn_channels)-1):
            conv_layer = nn.Conv2d(cnn_channels[i], cnn_channels[i+1],kernel_sizes[i], padding=paddings[i])
            bn_layer = nn.BatchNorm2d(cnn_channels[i+1]) if regularization == "bn" else nn.Identity()
            relu_layer = nn.ReLU()
            pool_layer = nn.MaxPool2d(pool_sizes[i])
            self.conv_layers.extend([conv_layer, bn_layer, relu_layer, pool_layer])

        # Calculate the final output dimensions after the convolutional layers
        test_input = torch.zeros((1, input_channel, img_height, img_width)) # batch_size=1, 1 channel, image width, image height
        with torch.no_grad():
            for layer in self.conv_layers:
                test_input = layer(test_input)
            self.conv_output_size = int(torch.prod(torch.tensor(test_input.shape)))

        # Define the fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_dims = [self.conv_output_size] + fc_dims
        for i in range(len(fc_dims) - 1):
            fc_layer = nn.Linear(fc_dims[i], fc_dims[i+1])
            relu_layer = nn.ReLU()
            dropout_layer = nn.Dropout(p=0.25) if regularization == "dropout" else nn.Identity()
            self.fc_layers.extend([fc_layer, relu_layer, dropout_layer])

        self.output_layer = nn.Linear(fc_dims[-1], out_dim)

    def forward(self, x):
        # Apply the convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, self.conv_output_size)
        
        # Apply the fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        x = self.output_layer(x)
        
        return x
    
class RegressionShrinkageLoss(nn.Module):
    def __init__(self, a=10.0, c=0.2):
        super(RegressionShrinkageLoss, self).__init__()
        self.a = a
        self.c = c

    def forward(self, predictions, targets):
        l1 = torch.abs(predictions - targets)
        l2 = l1**2
        loss = torch.mean((l2 / (1 + torch.exp(self.a * (self.c - l1)))).sum(1), 0)
        
        return loss
        # return torch.mean(l2 / 1 + torch.exp(self.a * (self.c - l1)))

    
    