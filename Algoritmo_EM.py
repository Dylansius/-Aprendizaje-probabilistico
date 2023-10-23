# Datos de ejemplo
data = [0.9, 1.2, 1.5, 1.8, 2.1]

# Inicializaci�n de los par�metros del modelo
mu = 1.0  # Media inicial
sigma = 1.0  # Desviaci�n est�ndar inicial

# N�mero de iteraciones EM
num_iteraciones = 10

for iteracion in range(num_iteraciones):
    # Expectation Step: Calcular las probabilidades posteriores
    posteriores = [1 / (sigma * (2 * 3.14159265358979323846) ** 0.5) * 2.718281828459045 ** (-0.5 * ((x - mu) / sigma) ** 2) for x in data]
    suma_posteriores = sum(posteriores)
    posteriores = [p / suma_posteriores for p in posteriores]

    # Maximization Step: Actualizar los par�metros del modelo
    suma_pesos = sum(p * x for p, x in zip(posteriores, data))
    suma_pesos_total = sum(posteriores)
    mu = suma_pesos / suma_pesos_total  # Actualizar la media
    suma_varianza_pesos = sum(p * ((x - mu) ** 2) for p, x in zip(posteriores, data))
    sigma = (suma_varianza_pesos / suma_pesos_total) ** 0.5  # Actualizar la desviaci�n est�ndar

    # Imprimir resultados de la iteraci�n actual
    print("Iteracion", iteracion + 1, ": mu =", round(mu, 3), ", sigma =", round(sigma, 3))

# Imprimir los par�metros finales estimados despu�s de todas las iteraciones
print("Resultado final: mu =", round(mu, 3), ", sigma =", round(sigma, 3))

