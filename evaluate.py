########## Lod dataset#########

import torch


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_data = datasets.ImageFolder("data/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=32)

class_names = test_data.classes
print("Classes:", class_names)





####### load model#######

"""
#resnet
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

model.load_state_dict(torch.load("resnet.pth"))
model = model.to(device)
model.eval()


#densenet
import torch.nn as nn
from torchvision.models import densenet121

model = densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 4)

model.load_state_dict(torch.load("densenet.pth"))
model = model.to(device)   
model.eval()
"""

#xception
import timm

model = timm.create_model('xception', pretrained=False, num_classes=4)

model.load_state_dict(torch.load("xception.pth"))

model = model.to(device)   
model.eval()
#####predictions######

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())


########Metrics########
print(classification_report(y_true, y_pred, target_names=class_names))

#######confusion matrix#######

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved!")