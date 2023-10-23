# Máquinas de Vectores Soporte (Núcleo) sin librerías

# Función para calcular el producto escalar entre dos vectores
def producto_escalar(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

# Función para predecir una instancia usando el clasificador SVM
def predecir(clasificador, instancia):
    decision_function = producto_escalar(clasificador.coef_, instancia) + clasificador.intercept_
    return clasificador.classes_[int(decision_function >= 0)]

# Datos de ejemplo (simplificados)
X_train = [[5.1, 3.5], [4.9, 3.0], [5.7, 2.8], [6.2, 2.9], [6.3, 3.3], [7.6, 3.0]]
y_train = [0, 0, 1, 1, 2, 2]

# Clasificador SVM con kernel lineal
class SVMClassifier:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = list(set(y))
        X_ = []
        for i in range(len(X)):
            X_.append([X[i], 1 if y[i] == self.classes_[0] else -1])
        self.coef_ = [0] * len(X_[0][0])
        lr = 0.01
        num_iter = 1000
        for _ in range(num_iter):
            for x, label in X_:
                if label * producto_escalar(self.coef_, x) < 1:
                    for j in range(len(self.coef_)):
                        self.coef_[j] += lr * (label * x[j] - 2 * 0.01 * self.coef_[j])
        self.intercept_ = 0

# Crear y entrenar el clasificador SVM
svm_classifier = SVMClassifier()
svm_classifier.fit(X_train, y_train)

# Datos de prueba
X_test = [[5.8, 2.7], [5.0, 3.5], [7.1, 3.0]]

# Realizar predicciones
y_pred = [predecir(svm_classifier, x) for x in X_test]

print("Predicciones:", y_pred)
