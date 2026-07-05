"""Generates a validation plot: given data points vs. the fitted curve."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

theta = np.deg2rad(30.0)
M = 0.03
X = 55.0

df = pd.read_csv("data/xy_data.csv")
x, y = df["x"].values, df["y"].values

t = np.linspace(6, 60, 2000)
E = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
xc = t * np.cos(theta) - E * np.sin(theta) + X
yc = 42 + t * np.sin(theta) + E * np.cos(theta)

plt.figure(figsize=(7, 6))
plt.scatter(x, y, s=8, alpha=0.4, label="given data points", color="tab:orange")
plt.plot(xc, yc, color="tab:blue", linewidth=1.5, label="fitted curve (θ=30°, M=0.03, X=55)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fitted parametric curve vs. given data")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("fit_validation.png", dpi=150)
print("saved fit_validation.png")