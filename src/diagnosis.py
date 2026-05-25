import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import timm
import matplotlib.pyplot as plt
import time

# ==========================================
# CONFIGURAÇÕES
# ==========================================

msInicio = int(time.time() * 1000)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('==================')
print(DEVICE)
print('==================')

TRAIN_DIR = '../processed-images/output/train'
VAL_DIR = '../processed-images/output/val'
TEST_DIR = '../processed-images/output/test'

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

# ==========================================
# TRANSFORMAÇÕES
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# DATASETS
# ==========================================

# TRAIN_DIR = 'spiral_dataset/train'
import os

print("Existe:", os.path.exists(TRAIN_DIR))

for root, dirs, files in os.walk(TRAIN_DIR):
    print(root)
    print("Dirs:", dirs)
    print("Files:", files[:5])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==========================================
# MODELO VISION TRANSFORMER
# ==========================================

model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=2
)

model = model.to(DEVICE)


# ==========================================
# FUNÇÃO DE PERDA E OTIMIZADOR
# ==========================================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# TREINAMENTO
# ==========================================
train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # =========================
    # VALIDAÇÃO
    # =========================

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{EPOCHS}]')
    print(f'Loss: {epoch_loss:.4f}')
    print(f'Val Accuracy: {accuracy:.2f}%')
    print('-' * 40)

# ==========================================
# AVALIAÇÃO FINAL
# ==========================================

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

print('\nREPORT CLASSIFICAÇÃO\n')
print(classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.classes
))

print('\nMATRIZ DE CONFUSÃO\n')
print(confusion_matrix(y_true, y_pred))

# ==========================================
# SALVAR MODELO
# ==========================================

torch.save(model.state_dict(), 'vit_parkinson_spiral.pth')

print('\nModelo salvo com sucesso!')

# ==========================================
# GRÁFICOS
# ==========================================

plt.figure(figsize=(10,4))
plt.plot(train_losses)
plt.title('Loss de Treinamento')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.figure(figsize=(10,4))
plt.plot(val_accuracies)
plt.title('Acurácia de Validação')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.show()

msFim = int(time.time() * 1000)

print("FIM")
print(f"Tempo de execução {(msFim-msInicio)/1000} segundos")