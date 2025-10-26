import numpy as np

"""
Direct Linear Transformation based on Förstner method (homogeneous system).
Inputs:
    R - (3, 3) rotation matrix
Returns:
    [omega, phi, kappa] angles in degree
"""

def rotation_matrix_to_angles(R):
    
    # check R is 3x3 matrix
    assert R.shape == (3, 3)

    # compute phi, omega, kappa angle
    phi = np.arcsin(R[0, 2])
    omega = -np.atan2(R[1, 2], R[2, 2])
    kappa = -np.atan2(R[0, 1], R[0, 0])

    # transfer from arc to deg
    phi   = np.degrees(phi)
    omega = np.degrees(omega)    
    kappa = np.degrees(kappa)

    # print(f"Omega (ω): {omega:.5f}°")
    # print(f"Phi (φ): {phi:.5f}°")
    # print(f"Kappa (κ): {kappa:.5f}°")

    return np.array([omega, phi, kappa], dtype=np.float64)

def as_euler(R_mat):
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_matrix(R_mat)

    # get euler angles in degree
    euler_angles = rotation.as_euler('zyx', degrees=True)
    np.set_printoptions(precision=8, suppress=True)
    omega = euler_angles[2]  
    phi   = euler_angles[1]   
    kappa = euler_angles[0] 

    # print(f"Omega (ω): {omega:.5f}°")
    # print(f"Phi (φ): {phi:.5f}°")
    # print(f"Kappa (κ): {kappa:.5f}°")

    return np.array([omega, phi, kappa], dtype=np.float64)