import os
import zipfile
import urllib.request

url = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset" # e.g., a Kaggle or Google Drive direct link
output = "brain_tumor_dataset.zip"

print("Downloading dataset...")
urllib.request.urlretrieve(url, output)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(".")
print("Done!")