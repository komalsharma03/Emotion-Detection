
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ========== Paths ==========
TRAIN_DIR = r"C:\Users\New User\OneDrive\Desktop\emotion\data\train"
TEST_DIR = r"C:\Users\New User\OneDrive\Desktop\emotion\data\test"
MODEL_PATH = r"C:\Users\New User\OneDrive\Desktop\emotion\code\emotion_CNN_model.pth"

# ========== Streamlit UI ==========
st.title("😊 Emotion Detection from Facial Expressions")
st.write("Upload an image or train the model to get started.")

# ========== Device Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# ========== Load Dataset ==========
if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = train_dataset.classes
    num_classes = len(class_names)
else:
    st.error("Train/Test dataset folder not found. Please check your paths.")
    st.stop()

# ========== Model Definition ==========
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ========== Load Model ==========
@st.cache_resource
def load_model():
    model = EmotionCNN(num_classes=num_classes).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model
    else:
        st.warning("Model not found. Please train the model first.")
        return None

model = load_model()

# ========== Training ==========
def train_model():
    model = EmotionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # Use more epochs for better performance
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    st.success("Model trained and saved successfully!")

# ========== Prediction ==========
def predict_emotion(image):
    if model is None:
        st.warning("Model is not loaded.")
        return None

    image = image.convert('L')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# ========== Buttons ==========
if st.button("Train Model"):
    train_model()
    st.cache_resource.clear()
    st.experimental_rerun()

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    emotion = predict_emotion(img)
    if emotion:
        st.success(f"Predicted Emotion: {emotion}")

if st.button("Visualize Test Predictions"):
    if model:
        model.eval()
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        for i in range(5):
            axes[i].imshow(images[i].squeeze().cpu(), cmap="gray")
            axes[i].set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
            axes[i].axis("off")
        st.pyplot(fig)

if st.button("Show Classification Report"):
    if model:
        y_true, y_pred = [], []
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        st.text("Classification Report:\n" + classification_report(y_true, y_pred, target_names=class_names))
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

