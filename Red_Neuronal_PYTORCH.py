import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# MNIST
print("Cargando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data / 255.0        
y = mnist.target.astype(int)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# arrays de NumPy a tensores de PyTorch

X_train_t = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# empaquetar imágenes y etiquetas juntas ( esto mantiene el orden )
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

#CNN

class CNN(nn.Module):


    def __init__(self):
        super(CNN, self).__init__()

        # --- BLOQUE CONVOLUCIONAL 1 ---
        self.conv1 = nn.Conv2d(
            in_channels=1,   
            out_channels=32, 
            kernel_size=3,   
            padding=1        
        )
        

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        # --- BLOQUE CONVOLUCIONAL 2 ---
        self.conv2 = nn.Conv2d(
            in_channels=32,  
            out_channels=64, 
            kernel_size=3,
            padding=1
        )
        

        # --- CAPAS TOTALMENTE CONECTADAS ---
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        

        self.fc2 = nn.Linear(128, 10)
        

        self.relu = nn.ReLU()
        

        self.dropout = nn.Dropout(0.25)
        

    def forward(self, x):
        

        x = self.relu(self.conv1(x))  
        x = self.pool(x)              

        x = self.relu(self.conv2(x))  
        x = self.pool(x)              

        x = x.view(x.size(0), -1)   
        

        x = self.dropout(x)           

        x = self.relu(self.fc1(x))    
        x = self.fc2(x)               
        

        return x

modelo = CNN()
print(modelo)
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

EPOCAS = 10
historial_loss = []

for epoca in range(EPOCAS):

    modelo.train()


    loss_total = 0

    for imagenes, etiquetas in train_loader:

        optimizador.zero_grad()

        predicciones = modelo(imagenes)

        loss = criterio(predicciones, etiquetas)

        loss.backward()
 
        optimizador.step()
        

        loss_total += loss.item()


    loss_media = loss_total / len(train_loader)
    historial_loss.append(loss_media)
    print(f"Época {epoca+1}/{EPOCAS} → Loss: {loss_media:.4f}")

    modelo.eval()


todas_preds  = []
todas_labels = []

with torch.no_grad():


    for imagenes, etiquetas in test_loader:
        predicciones = modelo(imagenes)     
        _, preds = torch.max(predicciones, 1)


        todas_preds.extend(preds.numpy())    
        todas_labels.extend(etiquetas.numpy())

acc = accuracy_score(todas_labels, todas_preds)
print(f"\n✅ Accuracy en test: {acc*100:.2f}%")


cm = confusion_matrix(todas_labels, todas_preds)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax)
ax.set_title(f"CNN | Accuracy: {acc*100:.2f}%")
plt.tight_layout()
plt.show()


filtros = modelo.conv1.weight.detach().numpy()

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
fig.suptitle("Los 32 filtros aprendidos por conv1 (cada uno detecta un patrón distinto)")

for i, ax in enumerate(axes.flat):
    ax.imshow(filtros[i, 0], cmap='viridis')
 
    ax.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCAS+1), historial_loss, marker='o', color='steelblue')
plt.title("Curva de aprendizaje: cómo bajó el error por época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()