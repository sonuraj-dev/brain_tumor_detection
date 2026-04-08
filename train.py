import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_data = datasets.ImageFolder("data/train", transform=train_transform)
val_data = datasets.ImageFolder("data/val", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

print("Classes:", train_data.classes)




"""

#Add ResNet Model

import torch.nn as nn
from torchvision import models

from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, 4)

model = model.to(device)




"""

"""

######       DenseNet Model         ###########
from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn

model = densenet121(weights=DenseNet121_Weights.DEFAULT)

# Modify classifier
model.classifier = nn.Linear(model.classifier.in_features, 4)

model = model.to(device)


"""
 


 #xception model




import timm
import torch.nn as nn

model = timm.create_model('xception', pretrained=True, num_classes=4)

model = model.to(device)

################Training function######



def train_model(model, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 🔹 VALIDATION
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")
train_model(model, epochs=20)
torch.save(model.state_dict(), "xception.pth")
print("Model saved!")