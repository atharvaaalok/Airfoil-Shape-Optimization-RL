import numpy as np
import Bezier_curves


class Bezier_spline:

    def __init__(self, ControlPoints_list):
        self.Bezier_curves = []
        self.total_curves = 0
        
        for ControlPoints in ControlPoints_list:
            if len(ControlPoints) == 3:
                curve_i = Bezier_curves.Quadratic_Bezier(*ControlPoints)
                self.Bezier_curves.append(curve_i)
                self.total_curves += 1
            elif len(ControlPoints) == 4:
                curve_i = Bezier_curves.Cubic_Bezier(*ControlPoints)
                self.Bezier_curves.append(curve_i)
                self.total_curves += 1

    def generate_points(self, total_points):
        t = np.linspace(0, self.total_curves, total_points)

        generated_points = []

        for i in range(self.total_curves):
            t_vals_curve_i = t[(t - i >= 0) & (t - i <= 1)] - i
            points_curve_i = self.Bezier_curves[i].generate_points_at_tvals(t_vals_curve_i)
            generated_points.append(points_curve_i)

        generated_points = np.vstack(generated_points)
        return generated_points