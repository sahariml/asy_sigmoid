import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def asymmetric_sigmoid(x, a, b):
    #return min_s + (max_s - min_s) * (1 - np.exp(-a * (x - b))) / (1 + np.exp(-a * (x - b)))
    return 0.1e1 / (0.1e1 / max_s + np.exp(-a * (x - b))) + 0.1e1 / (0.1e1 / min_s + np.exp(a * (x - b)))
#def asymmetric_sigmoid(x, a, b):
#    return min_s + (min_s - max_s) / (1 + np.exp(-a * (x - b)))

def objective_function(x, a, b):
    y_p1 = sigma_1 * min_s
    y_p2 = sigma_2 * max_s

    p1_residual = y_p1 - asymmetric_sigmoid(x_p1, a, b)
    p2_residual = y_p2 - asymmetric_sigmoid(x_p2, a, b)

    return np.array([p1_residual, p2_residual])

# Paramètres spécifiés
min_s = 0.8
max_s = 1.9
sigma_1 = 1.1  # 10%
sigma_2 = 0.9  # 90%
x_p1 = -10
x_p2 = 10

# Paramètres initiaux pour l'ajustement
initial_guess = [1.0, 0.0]

# Optimisation des paramètres a et b
optimized_params, _ = curve_fit(
    asymmetric_sigmoid, 
    np.array([x_p1, x_p2]), 
    np.array([sigma_1 * min_s, sigma_2 * max_s]),
    p0=initial_guess
)

# Affichage des paramètres optimaux
a_optimal, b_optimal = optimized_params
print(f"Optimal a: {a_optimal}")
print(f"Optimal b: {b_optimal}")

# Tracé de la fonction sigmoid asymétrique et des points spécifiés
x_values = np.linspace(-20, 20, 100)
y_values = asymmetric_sigmoid(x_values, a_optimal, b_optimal)

plt.plot(x_values, y_values, label='Asymmetric Sigmoid')
plt.axhline(y=min_s, color='r', linestyle='--', label='min_s')
plt.axhline(y=max_s, color='g', linestyle='--', label='max_s')
plt.scatter([x_p1, x_p2], [sigma_1 * min_s, sigma_2 * max_s], color='b', label='Points spécifiés')
plt.title('Asymmetric Sigmoid avec ajustement')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
