import pandas as pd
# Creación del dataframe
datos = {
    'Nombre': ['Juan', 'María', 'Pedro', 'Ana'],
    'Nota' : [7.5, 8.0, 4.0, 9.0],
    'Edad' : [28, 34, 22, 19]
}
df = pd.DataFrame(datos)
print(df)
# Columna ''aprobado''
df['Aprobado'] = df['Nota'] >= 5.0
print("\nDataFrame con columna 'Aprobado':\n", df)
print(df) 
#filtro por aprobados
aprobados = df[df['Aprobado'] == True]
print("\nEstudiantes aprobados:\n", aprobados)
#promedio notas
promedio = df['Nota'].mean()
print("\nPromedio de notas:", promedio)