from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def ResNet50():
    # Initialize model, loss function, and optimizer
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(
        *list(model.children())[:-2],  # Remove the fully connected layer and avgpool
        nn.Conv2d(2048, 256, kernel_size=3, padding=1),  # Adjust input channels to match ResNet50 output
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 2, kernel_size=3, padding=1),  # Change output channels to 2
        nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
    )
    return model