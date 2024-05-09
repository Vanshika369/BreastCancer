# %%
import torchvision
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from torchvision import transforms
from tqdm.notebook import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import os
import sys
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from matplotlib.image import imread

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor

# %%
current_working_directory = os.getcwd()
print(current_working_directory)

# %%
csv_path = "/scratch/aruna/ashwin/MG_Dataset/csv/meta.csv"
df_meta = pd.read_csv(csv_path)

# %%
dicom_data = pd.read_csv(
    "/scratch/aruna/ashwin/MG_Dataset/csv/dicom_info.csv")

print(dicom_data.head(10))

# %%
dicom_data.SeriesDescription.unique()

# %%
image_dir = "/scratch/aruna/ashwin/MG_Dataset/jpeg"
full_mammogram_images = dicom_data[dicom_data.SeriesDescription ==
                                   'full mammogram images'].image_path
cropped_images = dicom_data[dicom_data.SeriesDescription ==
                            'cropped images'].image_path
roi_mask_images = dicom_data[dicom_data.SeriesDescription ==
                             'ROI mask images'].image_path

full_mammogram_images = full_mammogram_images.apply(
    lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
cropped_images = cropped_images.apply(
    lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
roi_mask_images = roi_mask_images.apply(
    lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
full_mammogram_images.iloc[0]

# %%
print(cropped_images.shape)

# %%
full_mammogram_dict = dict()
cropped_dict = dict()
roi_mask_dict = dict()

for dicom in full_mammogram_images:
    key = dicom.split("/")[6]
    full_mammogram_dict[key] = dicom

for dicom in cropped_images:
    key = dicom.split("/")[6]
    cropped_dict[key] = dicom

for dicom in roi_mask_images:
    key = dicom.split("/")[6]
    roi_mask_dict[key] = dicom

# %%
mass_train_data = pd.read_csv(
    "/scratch/aruna/ashwin/MG_Dataset/csv/mass_case_description_train_set.csv")
mass_test_data = pd.read_csv(
    "/scratch/aruna/ashwin/MG_Dataset/csv/mass_case_description_test_set.csv")
calc_train_data = pd.read_csv(
    "/scratch/aruna/ashwin/MG_Dataset/csv/calc_case_description_train_set.csv")
calc_test_data = pd.read_csv(
    "/scratch/aruna/ashwin/MG_Dataset/csv/calc_case_description_test_set.csv")

# %%


def fix_image_path_mass(dataset):
    for i, img in enumerate(dataset.values):
        img_name = img[11].split("/")[2]
        if img_name in full_mammogram_dict:
            dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        img_name = img[12].split("/")[2]
        if img_name in cropped_dict:
            dataset.iloc[i, 12] = cropped_dict[img_name]

        img_name = img[13].split("/")[2]
        if img_name in roi_mask_dict:
            dataset.iloc[i, 13] = roi_mask_dict[img_name]


def fix_image_path_calc(dataset):
    for i, img in enumerate(dataset.values):
        img_name = img[11].split("/")[2]
        if img_name in full_mammogram_dict:
            dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        img_name = img[12].split("/")[2]
        if img_name in cropped_dict:
            dataset.iloc[i, 12] = cropped_dict[img_name]

        img_name = img[13].split("/")[2]
        if img_name in roi_mask_dict:
            dataset.iloc[i, 13] = roi_mask_dict[img_name]


# %%
fix_image_path_mass(mass_train_data)
fix_image_path_mass(mass_test_data)
fix_image_path_calc(calc_test_data)
fix_image_path_calc(calc_train_data)

# %%
train_data = pd.concat([mass_train_data, calc_train_data], ignore_index=True)
test_data = pd.concat([mass_test_data, calc_test_data], ignore_index=True)

# %%
print(test_data.shape)
print(train_data.shape)

# %%
test_data.columns

# %%
train_data = train_data[['cropped image file path', 'pathology']]
test_data = test_data[['cropped image file path', 'pathology']]

# %%
train_data = train_data.rename(columns={
    'cropped image file path': 'path',
})

test_data = test_data.rename(columns={
    'cropped image file path': 'path',
})

# %%
data = pd.concat([train_data, test_data], ignore_index=True)

# %%
data.pathology.unique()

# %%
data['pathology'].value_counts()

# %%
data['pathology'] = data['pathology'].replace(
    'BENIGN_WITHOUT_CALLBACK', 'BENIGN')

# %%
data['pathology'].value_counts()

# %%
image = imread(data[data['pathology'] == 'BENIGN']['path'].iloc[0])

plt.imshow(image, cmap='gray')
plt.title('BENIGN Image')
plt.axis('off')
plt.show()

# %%
print(data.shape)

# %%


def filter_dataframe_by_base_directory(df):
    base_directory = '/scratch/aruna/ashwin/MG_Dataset/jpeg/'

    # Check if all three columns start with the base directory
    mask = df['path'].str.startswith(base_directory)

    # Keep only the rows where all three columns start with the base directory
    filtered_df = df[mask]

    return filtered_df


# %%
data = filter_dataframe_by_base_directory(data)

# %%
print(data.shape)

# %%

label_encoder = LabelEncoder()
data['pathology'] = label_encoder.fit_transform(data['pathology'])

# %%

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Function to load and preprocess the image


def load_and_preprocess_image_with_progress(file_path):
    # Read image using PIL
    img = Image.open(file_path)

    # Apply transformations
    img = transform(img)

    return img


# Apply the function to the 'path' column
with tqdm(total=len(data)) as pbar:
    image_list = []
    for file_path in data['path']:
        image_list.append(load_and_preprocess_image_with_progress(file_path))
        pbar.update(1)
    data['image'] = image_list

# %%
print(data)

print("data['image'][0].shape : ", data['image'][0].shape)

# %%
data['pathology'].value_counts()

# %%
train_data.sample()

# %%

X = data['image'].values.reshape(-1, 1)
y = data['pathology']

oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

balanced_data = pd.DataFrame(
    {'image': X_resampled.flatten(), 'pathology': y_resampled})

# %%
balanced_data['pathology'].value_counts()

# %%

num_samples_to_visualize = 10
subset_data = balanced_data.head(num_samples_to_visualize)

fig, axes = plt.subplots(1, num_samples_to_visualize, figsize=(15, 5))

for i, (_, row) in enumerate(subset_data.iterrows()):
    image_tensor = row['image']
    image_tensor = image_tensor.cpu() if image_tensor.device.type == 'cuda' else image_tensor
    image_array = torchvision.transforms.ToPILImage()(image_tensor)

    axes[i].imshow(image_array, cmap='gray')
    axes[i].axis('off')

plt.show()

# %%

train_data, temp_data = train_test_split(
    balanced_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, random_state=42)

# %%
print("Training set size:", len(train_data))
print("Validation set size:", len(val_data))
print("Test set size:", len(test_data))

# %%


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx]['image']
        label = self.data.iloc[idx]['pathology']

        if self.transform:
            image = self.transform(image)

        return image, label


# %%
train_dataset = CustomDataset(train_data, transform=None)
val_dataset = CustomDataset(val_data, transform=None)
test_dataset = CustomDataset(test_data, transform=None)

# %%
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# %%
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.to(device)

# %%
num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        print(images.shape)
        images, labels = images.to(device), labels.to(
            device)  # Move inputs and targets to GPU

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(
                device)  # Move inputs and targets to GPU

            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Val Accuracy: {val_accuracy:.4f}')

    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_loss = val_loss

print('Training completed!')

# %%
# Define testing loop
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing', leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        test_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs.logits, 1)
        correct += (predicted == labels).sum().item()

        total += labels.size(0)

# Calculate average loss and accuracy
test_loss /= len(test_loader.dataset)
test_accuracy = correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
