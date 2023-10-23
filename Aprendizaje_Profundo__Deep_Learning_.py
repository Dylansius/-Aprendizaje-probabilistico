# Aprendizaje Profundo (Deep Learning) 

# Funcion para inicializar pesos y sesgos de una capa
def inicializar_parametros(input_size, output_size):
    W = 0.1  # Inicializacion de pesos
    b = 0.0  # Inicializacion de sesgo
    return W, b

# Funcion de activacion lineal
def activacion_lineal(W, b, X):
    return [W * x + b for x in X]

# Funcion de perdida (error cuadratico medio)
def calcular_perdida(Y_pred, Y_real):
    return sum((yp - yr) ** 2 for yp, yr in zip(Y_pred, Y_real)) / len(Y_pred)

# Funcion de optimizacion (descenso de gradiente)
def optimizar(X, Y_real, W, b, learning_rate, num_epochs):
    for _ in range(num_epochs):
        Y_pred = activacion_lineal(W, b, X)
        perdida = calcular_perdida(Y_pred, Y_real)
        dW = sum(2 * (yp - yr) * x for yp, yr, x in zip(Y_pred, Y_real, X)) / len(Y_pred)
        db = sum(2 * (yp - yr) for yp, yr in zip(Y_pred, Y_real)) / len(Y_pred)
        W -= learning_rate * dW
        b -= learning_rate * db
    return W, b

# Datos de ejemplo
X = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Y_real = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test = X[:8], X[8:]
y_train, y_test = Y_real[:8], Y_real[8:]

# Inicializacion de parametros
W, b = inicializar_parametros(1, 1)

# Hiperparametros
learning_rate = 0.1
num_epochs = 100

# Entrenamiento
W, b = optimizar(X_train, y_train, W, b, learning_rate, num_epochs)

# Evaluacion en los datos de prueba
Y_pred = activacion_lineal(W, b, X_test)
perdida_test = calcular_perdida(Y_pred, y_test)
print("Perdida en datos de prueba:", perdida_test)

# Hacemos predicciones
print("Predicciones:")
for i in range(len(X_test)):
    print(f"Entrada: {X_test[i]}, Prediccion: {Y_pred[i]}, Real: {y_test[i]}")
