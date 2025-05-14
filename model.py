import torch
import torchvision
from torch import nn

def create_model(num_classes: int = 2):
    # Load pre-trained ResNet50 weights and associated transforms
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    transforms = weights.transforms()

    # Load the pre-trained ResNet50 model
    model = torchvision.models.resnet50(weights=weights)

    # Freeze all layers of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Replace the final fully connected layer to fit the number of classes
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 500),
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=500, out_features=num_classes, bias=True)  # Use num_classes here
    )

    return model, transforms
