import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. Define the true function
def f(x):
    return np.sin(x)

# 2. Sample some training data (with noise)
X_train = np.linspace(0.1, 9.9, 10).reshape(-1, 1)
y_train = f(X_train).ravel() + 0.1 * np.random.randn(len(X_train))

# 3. Define the kernel and the GPR model
kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01**2, n_restarts_optimizer=10)

# 4. Fit the model
gpr.fit(X_train, y_train)

# 5. Predict
X_test = np.linspace(0, 10, 1000).reshape(-1, 1)
y_pred, sigma = gpr.predict(X_test, return_std=True)

# 6. Plot
plt.figure(figsize=(10, 5))
plt.plot(X_test, f(X_test), 'r--', label="True function $f(x)=\sin(x)$")
plt.scatter(X_train, y_train, c='k', label="Noisy observations")
plt.plot(X_test, y_pred, 'b', label="GPR prediction")
plt.fill_between(X_test.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma,
                 alpha=0.2, color='blue', label='95% confidence interval')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title("Gaussian Process Regression")
plt.grid(True)
plt.show()
