import torch.nn as nn
from torchvision.models import vgg19
import torch

class MyVGG19(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.layer_1 = self.make_block_1(3, 64)     # 64, 112, 112
        self.layer_2 = self.make_block_1(64, 128)   # 128, 56, 56
        self.layer_3 = self.make_block_2(128, 256)  # 256, 28, 28
        self.layer_4 = self.make_block_2(256, 512)  # 512, 14, 14
        self.layer_5 = self.make_block_2(512, 512)  # 512, 7, 7
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc_1 = self.make_fc(25088, 4096)
        self.fc_2 = self.make_fc(4096, 4096)
        self.fc_3 = nn.Linear(in_features=4096, out_features=num_classes)

    def make_block_1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
    
    def make_block_2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
    
    def make_fc(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x

if __name__ == "__main__":
    model = MyVGG19(num_classes=5)
    fake_image = torch.rand(1, 3, 224, 224)
    output = model(fake_image)
    print(output.shape)
