import math, numpy as np, pandas as pd

# ==========================================================
# 1. Fungsi utama sistem persamaan
# ==========================================================
def f(xy):
    x, y = xy
    return np.array([x*x + x*y - 10, y + 3*x*y*y - 57])

# Jacobian (turunan parsial untuk Newton-Raphson)
def J(xy):
    x, y = xy
    return np.array([[2*x + y, x],
                     [3*y*y, 1 + 6*x*y]])

# ==========================================================
# 2. Fungsi Iterasi Titik Tetap (g1A, g2A)
# ==========================================================
def g1A(x, y):
    val = 10 - x*y
    return math.sqrt(val) if val >= 0 else float('nan')

def g2A(x_next, y):
    val = (57 - y) / (3 * x_next)
    return math.sqrt(val) if val >= 0 else float('nan')

# ==========================================================
# 3. Metode Iterasi Titik Tetap (Jacobi / Seidel)
# ==========================================================
def iterate_IT(method="seidel", x0=1.5, y0=3.5, eps=1e-6, maxiter=100):
    x, y = x0, y0
    data = []
    for r in range(1, maxiter+1):
        x_new = g1A(x, y)
        y_new = g2A(x_new if method=="seidel" else x, y)
        dx, dy = abs(x_new-x), abs(y_new-y)
        data.append((r, x_new, y_new, dx, dy))
        if dx < eps and dy < eps:
            break
        x, y = x_new, y_new
    df = pd.DataFrame(data, columns=["Iter", "x", "y", "Δx", "Δy"])
    return df, (x, y)

# ==========================================================
# 4. Metode Newton-Raphson
# ==========================================================
def newton_raphson(x0=1.5, y0=3.5, eps=1e-6, maxiter=100):
    x, y = x0, y0
    data = []
    for r in range(1, maxiter+1):
        step = np.linalg.solve(J((x,y)), -f((x,y)))
        x_new, y_new = x+step[0], y+step[1]
        dx, dy = abs(x_new-x), abs(y_new-y)
        data.append((r, x_new, y_new, dx, dy))
        if dx < eps and dy < eps:
            break
        x, y = x_new, y_new
    df = pd.DataFrame(data, columns=["Iter", "x", "y", "Δx", "Δy"])
    return df, (x, y)

# ==========================================================
# 5. Metode Secant (Broyden Good)
# ==========================================================
def broyden_good(x0=1.5, y0=3.5, eps=1e-6, maxiter=100):
    x, y = x0, y0
    B = J((x, y))
    F = f((x, y))
    data = []
    for r in range(1, maxiter+1):
        step = np.linalg.solve(B, -F)
        x_new, y_new = x+step[0], y+step[1]
        dx, dy = abs(x_new-x), abs(y_new-y)
        data.append((r, x_new, y_new, dx, dy))
        if dx < eps and dy < eps:
            break
        s = np.array([x_new-x, y_new-y])
        yv = f((x_new, y_new)) - F
        B = B + np.outer((yv - B@s), s) / (s@s)
        x, y, F = x_new, y_new, f((x_new, y_new))
    df = pd.DataFrame(data, columns=["Iter", "x", "y", "Δx", "Δy"])
    return df, (x, y)

# ==========================================================
# 6. Jalankan semua metode
# ==========================================================
df_jacobi, root_jacobi = iterate_IT("jacobi")
df_seidel, root_seidel = iterate_IT("seidel")
df_newton, root_newton = newton_raphson()
df_secant, root_secant = broyden_good()

# ==========================================================
# 7. Tampilkan hasil
# ==========================================================
print("\n========== HASIL ITERASI TITIK TETAP (JACOBI) ==========")
print(df_jacobi.head(10).to_string(index=False))
print(f"\nHasil akhir Jacobi: x = {root_jacobi[0]:.6f}, y = {root_jacobi[1]:.6f}\n")

print("========== HASIL ITERASI TITIK TETAP (SEIDEL) ==========")
print(df_seidel.head(10).to_string(index=False))
print(f"\nHasil akhir Seidel: x = {root_seidel[0]:.6f}, y = {root_seidel[1]:.6f}\n")

print("========== HASIL NEWTON–RAPHSON ==========")
print(df_newton.to_string(index=False))
print(f"\nHasil akhir Newton–Raphson: x = {root_newton[0]:.6f}, y = {root_newton[1]:.6f}\n")

print("========== HASIL SECANT (BROYDEN GOOD) ==========")
print(df_secant.to_string(index=False))
print(f"\nHasil akhir Secant (Broyden): x = {root_secant[0]:.6f}, y = {root_secant[1]:.6f}\n")

# ==========================================================
# 8. Ringkasan singkat
# ==========================================================
print("========== RINGKASAN AKHIR ==========")
summary = pd.DataFrame([
    ["IT – Jacobi (g1A,g2A)", root_jacobi[0], root_jacobi[1], len(df_jacobi)],
    ["IT – Seidel (g1A,g2A)", root_seidel[0], root_seidel[1], len(df_seidel)],
    ["Newton–Raphson", root_newton[0], root_newton[1], len(df_newton)],
    ["Secant (Broyden)", root_secant[0], root_secant[1], len(df_secant)]
], columns=["Metode", "x*", "y*", "Iterasi"])
print(summary.to_string(index=False))
