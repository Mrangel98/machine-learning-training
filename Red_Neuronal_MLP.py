# ============================================================
# imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# cargando el dataset
# ============================================================
print("Cargando MNIST... (puede tardar un momento la primera vez)")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data              
y = mnist.target.astype(int) 

print(f"Forma de X: {X.shape}")  
print(f"Forma de y: {y.shape}")  

# ============================================================
# normlizacion
# ============================================================
X = X / 255.0

# ============================================================
# division prueba-test (80/20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")

# ============================================================
# configuraciones diferentes para comparar
# ============================================================

configuraciones = {
    "Config 1 - Simple": MLPClassifier(
        hidden_layer_sizes=(128,),        # 1 capa oculta 
        activation='relu',               
        solver='adam',                    
        max_iter=20,
        random_state=42,
        verbose=True
    ),
    "Config 2 - Media": MLPClassifier(
        hidden_layer_sizes=(256, 128),    # 2 capas ocultas
        activation='relu',
        solver='adam',
        max_iter=20,
        random_state=42,
        verbose=True
    ),
    "Config 3 - Profunda": MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # 3 capas ocultas
        activation='tanh',                   
        solver='sgd',                        
        learning_rate_init=0.01,
        max_iter=20,
        random_state=42,
        verbose=True
    ),
}

resultados = {}

for nombre, modelo in configuraciones.items():
    print(f"\n{'='*50}")
    print(f"Entrenando: {nombre}")
    print(f"{'='*50}")
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    resultados[nombre] = {"modelo": modelo, "y_pred": y_pred, "accuracy": acc}
    print(f"✅ Accuracy en test: {acc:.4f} ({acc*100:.2f}%)")

# ============================================================
# visualizacion de los resultados para compaarar
# ============================================================


for nombre, datos in resultados.items():
    cm = confusion_matrix(y_test, datos["y_pred"])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(f"Matriz de Confusión\n{nombre} | Accuracy: {datos['accuracy']*100:.2f}%")
    plt.tight_layout()
    plt.show()


print("\n📊 RESUMEN DE RESULTADOS")
print("="*45)
for nombre, datos in resultados.items():
    print(f"{nombre}: {datos['accuracy']*100:.2f}%")