import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Carga el dataset iris
iris = load_iris()

# se pasa a Dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["especie"] = iris.target  # 0=Setosa, 1=Versicolor, 2=Virginica

print("=== Primeras filas del dataset ===")
print(df.head())

print("\n=== Información general ===")
print(df.info())

print("\n=== Clases disponibles ===")
print(iris.target_names)

print("\n=== Distribución de clases ===")
print(df["especie"].value_counts())

# separacion y test
X = iris.data
y = iris.target

# Separamos: 80% entrenamiento, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de test: {X_test.shape[0]} muestras")

scaler = StandardScaler()

# Ajustamos y transformamos el conjunto de entrenamiento
X_train = scaler.fit_transform(X_train)

# Solo transformamos el test (nunca ajustamos con datos de test)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predecimos sobre el conjunto de test
y_pred = knn.predict(X_test)

# Evaluamos
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Resultados con K=5 ===")
print(f"Accuracy: {accuracy:.2%}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Matriz de confusión - K=5")
plt.show()

for k in [1, 15]:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"K={k} → Accuracy: {acc:.2%}")

    valores_k = range(1, 21)
accuracies = []

for k in valores_k:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    accuracies.append(accuracy_score(y_test, pred))

plt.figure(figsize=(10, 5))
plt.plot(valores_k, accuracies, marker='o', color='steelblue')
plt.title("Accuracy según valor de K")
plt.xlabel("Valor de K")
plt.ylabel("Accuracy")
plt.xticks(valores_k)
plt.grid(True)
plt.show()