import numpy as np


class Quadratic_Bezier:

    Characteristic_mat = np.array([[1, 0, 0], [-2, 2, 0], [1, -2, 1]])

    def __init__(self, P0, P1, P2):
        self.P0 = np.array(P0)
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
        self.ControlPoints_mat = np.vstack((self.P0, self.P1, self.P2))
    
    def generate_points(self, total_points):
        t = np.linspace(0, 1, total_points)
        T_mat = np.vstack((t ** 0, t, t ** 2)).T
        generated_points = T_mat @ Quadratic_Bezier.Characteristic_mat @ self.ControlPoints_mat
        return generated_points
    
    def generate_points_at_tvals(self, t):
        T_mat = np.vstack((t ** 0, t, t ** 2)).T
        generated_points = T_mat @ Quadratic_Bezier.Characteristic_mat @ self.ControlPoints_mat
        return generated_points


class Cubic_Bezier:

    Characteristic_mat = np.array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])

    def __init__(self, P0, P1, P2, P3):
        self.P0 = np.array(P0)
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
        self.P3 = np.array(P3)
        self.ControlPoints_mat = np.vstack((self.P0, self.P1, self.P2, self.P3))
    
    def generate_points(self, total_points):
        t = np.linspace(0, 1, total_points)
        T_mat = np.vstack((t ** 0, t, t ** 2, t ** 3)).T
        generated_points = T_mat @ Cubic_Bezier.Characteristic_mat @ self.ControlPoints_mat
        return generated_points

    def generate_points_at_tvals(self, t):
        T_mat = np.vstack((t ** 0, t, t ** 2, t ** 3)).T
        generated_points = T_mat @ Cubic_Bezier.Characteristic_mat @ self.ControlPoints_mat
        return generated_points