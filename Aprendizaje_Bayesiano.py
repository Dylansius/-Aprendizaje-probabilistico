#Aprendizaje Bayesiano

# Datos observados (lanzamientos de una moneda)
data = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1]

# Estimación inicial de la probabilidad de cara (prior)
probabilidad_cara = 0.5

# Número de lanzamientos
total_lanzamientos = len(data)

# Contar la cantidad de caras
caras = sum(data)

# Parámetros previos Beta para la distribución a priori
alpha_prior = 1
beta_prior = 1

# Actualización de la estimación bayesiana de la probabilidad de cara
alpha_posterior = alpha_prior + caras
beta_posterior = beta_prior + total_lanzamientos - caras

# Estimar la probabilidad de cara posterior
probabilidad_cara_posterior = alpha_posterior / (alpha_posterior + beta_posterior)

print(f"Probabilidad de cara estimada: {probabilidad_cara_posterior:.2f}")
