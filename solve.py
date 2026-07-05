"""
theta, M, X for the parametric curve

    x = t*cos(theta) - e^(M|t|)*sin(0.3t)*sin(theta) + X
    y = 42 + t*sin(theta) + e^(M|t|)*sin(0.3t)*cos(theta)

from a cloud of (x, y) samples with t in (6, 60).

Idea: the two equations are a 2D rotation of the vector (t, E) --
where E = e^(M|t|)*sin(0.3t) -- by angle theta, shifted by (X, 42).
Rotations are invertible, so for ANY candidate (theta, X) we can solve
for t and E for every data point in closed form:

    t = (x - X)*cos(theta) + (y - 42)*sin(theta)
    E = (y - 42)*cos(theta) - (x - X)*sin(theta)

This turns the fitting problem from "1500 unknown t_i's + 3 params"
into a plain 3-parameter nonlinear least squares problem on the
residual  E_i - e^(M*|t_i|)*sin(0.3*t_i).
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
DATA_PATH = "data/xy_data.csv"
THETA_BOUNDS_DEG = (0, 50)
M_BOUNDS = (-0.05, 0.05)
X_BOUNDS = (0, 100)
N_RESTARTS = 60
SEED = 0
def residuals(params, x, y):
    theta, M, X = params
    ct, st = np.cos(theta), np.sin(theta)
    u = x - X
    v = y - 42.0
    t = u * ct + v * st
    E = v * ct - u * st
    E_pred = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    return E_pred - E
def fit(x, y):
    theta_lo, theta_hi = np.deg2rad(THETA_BOUNDS_DEG[0]), np.deg2rad(THETA_BOUNDS_DEG[1])
    m_lo, m_hi = M_BOUNDS
    xshift_lo, xshift_hi = X_BOUNDS
    rng = np.random.default_rng(SEED)
    best_cost, best_params = np.inf, None
    for _ in range(N_RESTARTS):
        x0 = [
            rng.uniform(theta_lo, theta_hi),
            rng.uniform(m_lo, m_hi),
            rng.uniform(xshift_lo, xshift_hi),
        ]
        sol = least_squares(
            residuals, x0=x0, args=(x, y),
            bounds=([theta_lo, m_lo, xshift_lo], [theta_hi, m_hi, xshift_hi]),
            method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15,
        )
        cost = float(np.sum(sol.fun ** 2))
        if cost < best_cost:
            best_cost, best_params = cost, sol.x
    return best_params, best_cost
def reconstruction_error(params, x, y):
    theta, M, X = params
    ct, st = np.cos(theta), np.sin(theta)
    u = x - X
    v = y - 42.0
    t = u * ct + v * st
    E = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x_pred = t * ct - E * st + X
    y_pred = 42 + t * st + E * ct
    err = np.abs(x_pred - x) + np.abs(y_pred - y)
    return err.mean(), err.max(), t.min(), t.max()
def main():
    df = pd.read_csv(DATA_PATH)
    x, y = df["x"].values, df["y"].values
    params, cost = fit(x, y)
    theta, M, X = params
    mean_l1, max_l1, t_min, t_max = reconstruction_error(params, x, y)
    print(f"theta = {np.rad2deg(theta):.4f} deg  ({theta:.6f} rad)")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.4f}")
    print(f"fit cost (sum sq residual): {cost:.3e}")
    print(f"mean L1 reconstruction error: {mean_l1:.3e}")
    print(f"max  L1 reconstruction error: {max_l1:.3e}")
    print(f"recovered t range: [{t_min:.3f}, {t_max:.3f}]")

if __name__ == "__main__":
    main()