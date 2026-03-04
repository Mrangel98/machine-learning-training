# ============================================================
# PASO 1: Importaciones
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# PASO 2: Cargar y preparar los datos
# ============================================================
print("Cargando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data / 255.0
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

print(f"Entrenamiento: {X_train.shape}")
print(f"Prueba:        {X_test.shape}")

# ============================================================
# PASO 3: Definir la arquitectura
# ============================================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

# ============================================================
# PASO 4: Compilar el modelo
# ============================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# PASO 5: Entrenar con TensorBoard
# ============================================================
tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[tensorboard_callback],
    verbose=1
)

# ============================================================
# PASO 6: Evaluar el modelo
# ============================================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Accuracy en test: {test_acc*100:.2f}%")
print(f"   Loss en test:     {test_loss:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, colorbar=True)
ax.set_title(f"CNN Keras | Accuracy: {test_acc*100:.2f}%")
plt.tight_layout()
plt.show()

# ============================================================
# PASO 7: Curvas de aprendizaje
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['loss'],     label='Entrenamiento', marker='o')
ax1.plot(history.history['val_loss'], label='Validación',    marker='o')
ax1.set_title('Curva de Loss')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['accuracy'],     label='Entrenamiento', marker='o')
ax2.plot(history.history['val_accuracy'], label='Validación',    marker='o')
ax2.set_title('Curva de Accuracy')
ax2.set_xlabel('Época')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# PASO 8: Abrir TensorBoard (ejecutar en terminal aparte)
# ============================================================
# python -m tensorboard.main --logdir=logs
# Luego abrir navegador en: http://localhost:6006