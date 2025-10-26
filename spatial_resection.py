import numpy as np
import copy
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, Matrix, simplify, diff, lambdify

# ---------------------- partial derivative ----------------------
def build_symbolic_jacobian():
    omega, phi, kappa = symbols('omega phi kappa')
    XL, YL, ZL = symbols('XL YL ZL')
    X, Y, Z = symbols('X Y Z')
    x0, y0, f = symbols('x0 y0 f')
    
    # Rotation matrices
    Rw = Matrix([[1, 0, 0],
                 [0, cos(omega), sin(omega)],
                 [0, -sin(omega), cos(omega)]])
    Rp = Matrix([[cos(phi), 0, -sin(phi)],
                 [0, 1, 0],
                 [sin(phi), 0, cos(phi)]])
    Rk = Matrix([[cos(kappa), sin(kappa), 0],
                 [-sin(kappa), cos(kappa), 0],
                 [0, 0, 1]])

    R = Rk * Rp * Rw
    dXYZ = Matrix([X - XL, Y - YL, Z - ZL])
    Xc, Yc, Zc = R * dXYZ

    x = x0 - f * (Xc / Zc)
    y = y0 - f * (Yc / Zc)

    vars = [omega, phi, kappa, XL, YL, ZL]
    dx_partials = [simplify(diff(x, v)) for v in vars]
    dy_partials = [simplify(diff(y, v)) for v in vars]

    all_partials = dx_partials + dy_partials
    symbols_all = ['omega', 'phi', 'kappa', 'XL', 'YL', 'ZL', 'X', 'Y', 'Z', 'x0', 'y0', 'f']

    funcs = [lambdify(symbols_all, expr, modules='numpy') for expr in all_partials]
    return funcs  # 12 derivative functions

# ---------------------- projection ----------------------
def project_point(iop, eop, X, Y, Z, delta):
    from math import sin, cos
    x0, y0, f = iop['x0'], iop['y0'], iop['f']
    omega, phi, kappa = np.radians(eop['omega']), np.radians(eop['phi']), np.radians(eop['kappa'])
    XL, YL, ZL = eop['XL'], eop['YL'], eop['ZL']+delta

    Rw = np.array([[1, 0, 0],
                   [0, cos(omega), sin(omega)],
                   [0, -sin(omega), cos(omega)]])
    Rp = np.array([[cos(phi), 0, -sin(phi)],
                   [0, 1, 0],
                   [sin(phi), 0, cos(phi)]])
    Rk = np.array([[cos(kappa), sin(kappa), 0],
                   [-sin(kappa), cos(kappa), 0],
                   [0, 0, 1]])

    R = Rk @ Rp @ Rw
    dXYZ = np.array([X - XL, Y - YL, Z - ZL])
    Xc, Yc, Zc = R @ dXYZ

    x = x0 - f * (Xc / Zc)
    y = y0 - f * (Yc / Zc)
    return x, y

# ---------------------- spatial resection ----------------------
def spatial_resection(iop, eop, object_points, image_points, delta=0):
    
    funcs = build_symbolic_jacobian()
    convergence_log = []

    for iteration in range(50):
        A_list = []
        L_list = []

        for (x_obs, y_obs), (X, Y, Z) in zip(image_points, object_points):
            x_hat, y_hat = project_point(iop, eop, X, Y, Z, delta)
            v = np.array([x_obs - x_hat, y_obs - y_hat])
            L_list.append(v)

            args = [
                np.radians(eop['omega']),
                np.radians(eop['phi']),
                np.radians(eop['kappa']),
                eop['XL'], eop['YL'], eop['ZL'] + delta,
                X, Y, Z, iop['x0'], iop['y0'], iop['f']]

            dx_partials = np.array([fn(*args) for fn in funcs[:6]])
            dy_partials = np.array([fn(*args) for fn in funcs[6:]])
            A = np.vstack([dx_partials, dy_partials])  
            A_list.append(A)

        A_stack = np.vstack(A_list)            
        L_stack = np.hstack(L_list).reshape(-1,1)

        N = A_stack.T @ A_stack
        u = A_stack.T @ L_stack

        try:
            dt = np.linalg.solve(N, u).flatten()
        except np.linalg.LinAlgError:
            print("⚠️ N is singular, using pseudo-inverse")
            dt = (np.linalg.pinv(N) @ u).flatten()

    
        for i,key in enumerate(['omega','phi','kappa','XL','YL','ZL']):
            eop[key] += dt[i]

        norm_delta = np.linalg.norm(dt)
        convergence_log.append((iteration + 1, norm_delta))
        if norm_delta < 1e-5:
            break

    vTPv = float(L_stack.T @ L_stack)
    dof = len(L_stack) - 6
    sigma0 = np.sqrt(vTPv / dof)
    cov = sigma0**2 * np.linalg.inv(N)
    print("\n=== Spatial Resection Estimated EOP ===")
    print(f"Focal Length f: {iop['f']:.6f} mm")
    print(f"Image Center x0: [{iop['x0']:.6f}, {iop['y0']:.6f}] mm")
    print(f"Omega, Phi, Kappa (deg) : [{eop['omega']:.6f}, {eop['phi']:.6f}, {eop['kappa']:.6f}]")
    print(f"Camera Position X (m) : [{eop['XL']/1000:.6f}, {eop['YL']/1000:.6f}, {eop['ZL']/1000:.6f}]")
    # for k in ['omega', 'phi', 'kappa']:
    #     print(f"{k}: {eop[k]:.6f}°")
    # for k in ['XL', 'YL', 'ZL']:
    #     print(f"{k}: {eop[k]:.3f} mm")

    print(f"\nSigma0: {sigma0:.4f} mm")

    # print("\nCovariance matrix of EOP:")
    # print(cov)

    print("\nConvergence log:")
    if convergence_log:
        last_it, last_err = convergence_log[-1]
        print(f" Final Iter {last_it:2d}: |delta| = {last_err:.6e}")

    iters = [it for it,_ in convergence_log]
    errors = [err for _,err in convergence_log]

    plt.figure()
    plt.plot(iters, errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('|delta| (norm)')
    plt.title('Spatial Resection Convergence')
    plt.grid(True)
    plt.show()
    
    return eop

