# AI R&D Assignment — Parametric Curve Fit
## Problem
Recover θ, M, X in:

```
x = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X
y = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```
given 1500 (x, y) samples for t ∈ (6, 60), with:

```
0°    < θ < 50°
-0.05 < M < 0.05
0     < X < 100
```
## Result

| Variable | Value |
|---|---|
| θ | 30° (π/6 rad ≈ 0.523599) |
| M | 0.03 |
| X | 55 |

Reconstruction error against the given data: mean L1 ≈ 3.5×10⁻⁶, max L1 ≈ 2.4×10⁻⁵ — effectively exact.

**Desmos graph:** https://www.desmos.com/calculator/YOUR-SAVED-ID

## Approach

The naive way to fit this is to treat every point's own `t` as unknown and search for the closest point on a candidate curve — a nested optimization that's slow and prone to getting stuck in local minima.

There's a cleaner way in, because of how the equations are built. Group the exponential/sine term as `E(t) = e^(M|t|)·sin(0.3t)`, and the equations become:

```
x − X = t·cos(θ) − E·sin(θ)
y − 42 = t·sin(θ) + E·cos(θ)
```

That's just the vector `(t, E)` rotated by angle θ, then shifted by `(X, 42)`. Rotation matrices are orthogonal, so this is trivially invertible — for *any* candidate `(θ, X)`, every data point's `t` and `E` can be recovered in closed form, with no search required:

```
t = (x − X)·cos(θ) + (y − 42)·sin(θ)
E = (y − 42)·cos(θ) − (x − X)·sin(θ)
```

This collapses the problem from "3 parameters + 1500 hidden t's" down to just 3 parameters. The remaining unknown, M, sits inside `E = e^(M|t|)·sin(0.3t)`, so the fitting objective becomes a per-point residual:

```
residual_i(θ, M, X) = e^(M|t_i|)·sin(0.3·t_i) − E_i
```
where `t_i` and `E_i` are the closed-form values above, both functions of θ and X only. This is now an ordinary 3-parameter nonlinear least-squares problem, solved with `scipy.optimize.least_squares` (Trust Region Reflective algorithm, bounded to the ranges given in the assignment).

To make sure the result isn't a local minimum, the solver runs from 60 random starting points sampled uniformly across the allowed (θ, M, X) ranges. All restarts converge to the same values (θ=30°, M=0.03, X=55) with a residual on the order of 1e-8 — strong evidence this is the true global solution rather than a coincidence of one lucky starting point.

## Files

-> `solve.py` — main script, recovers (θ, M, X) from `data/xy_data.csv`
-> `plot_fit.py` — overlays the fitted curve on the given data points for a visual sanity check (`fit_validation.png`)
-> `data/xy_data.csv` — the given data (copied in for reproducibility)
-> `requirements.txt` — numpy, pandas, scipy, matplotlib

## Running it

```bash
pip install -r requirements.txt
python solve.py       # prints theta, M, X
python plot_fit.py    # produces fit_validation.png
```
## Desmos steps (how the graph was built)

1. Open the given calculator: https://www.desmos.com/calculator/rfj91yrxob
2. Add a new expression with the solved values substituted in (θ in radians):

```
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)

```
3. Set the domain restriction to `6 ≤ t ≤ 60`.
4. Click **Save Copy** to get a personal shareable link.

## References

The nonlinear least-squares fit in `solve.py` uses SciPy's `least_squares` function with the Trust Region Reflective (`'trf'`) method, which implements the subspace/interior-point approach described in:

Branch, M. A., Coleman, T. F., & Li, Y. (1999). A subspace, interior, and conjugate gradient method for large-scale bound-constrained minimization problems. *SIAM Journal on Scientific Computing, 21*(1), 1–23. 

The general trust-region optimization framework this method builds on is also covered in:

Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2nd ed., Chapter 4). Springer.

All numerical fitting was performed using SciPy:

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., ... SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*(3), 261–272. 

## Final answer

```
θ = 30°  (π/6 rad)
M = 0.03
X = 55
```