import streamlit as st
import numpy as np
from scipy import linalg
from math import ceil, pi  # Importing ceil and pi functions
import matplotlib.pyplot as plt

def lowess(x, y, f, iterations):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n) 
    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i] 
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    return yest

# Streamlit app
st.title('Lowess Regression Implementation')
st.write('This app performs Locally Weighted Scatterplot Smoothing (Lowess) on generated data.')

# Sidebar for user input
st.sidebar.header('User Input Parameters')
n = st.sidebar.slider('Number of data points', 10, 500, 100)
f = st.sidebar.slider('Smoothing parameter (f)', 0.01, 1.0, 0.25)
iterations = st.sidebar.slider('Number of iterations', 1, 10, 3)
noise = st.sidebar.slider('Noise level', 0.0, 1.0, 0.3)

# Generate data
x = np.linspace(0, 2 * pi, n)
y = np.sin(x) + noise * np.random.randn(n)

# Apply Lowess
yest = lowess(x, y, f, iterations)

# Plotting
st.subheader('Generated Data vs Lowess Estimate')
fig, ax = plt.subplots()
ax.plot(x, y, 'r.', label='Noisy data')
ax.plot(x, yest, 'b-', label='Lowess estimate')
ax.legend()
st.pyplot(fig)

st.write('Adjust the parameters to see how the Lowess algorithm fits the data.')
