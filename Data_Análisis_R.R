library(dplyr)

# 1. Creamos el DataFrame con datos de productos
df <- data.frame(
  Nombre    = c("Laptop", "Camiseta", "Smartphone", "Pantalón", "Auriculares"),
  Categoria = c("Electrónica", "Ropa", "Electrónica", "Ropa", "Electrónica"),
  Precio    = c(999.99, 19.99, 599.99, 39.99, 149.99)
)

print("=== DataFrame original ===")
print(df)
# 2. Filtramos los productos de la categoría "Electrónica"
electronica <- filter(df, Categoria == "Electrónica")

print("=== Productos de Electrónica ===")
print(electronica)
# 3. Calculamos el precio promedio de los productos de Electrónica
promedio <- mean(electronica$Precio)

cat("=== Precio promedio de Electrónica: ", promedio, "€ ===\n")
