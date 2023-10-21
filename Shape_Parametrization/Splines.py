import numpy as np
import Curves


class Bezier_spline:

    def __init__(self, ControlPoints_list):
        self.Bezier_curves = []
        self.total_curves = len(ControlPoints_list)
        
        for ControlPoints in ControlPoints_list:
            if len(ControlPoints) == 3:
                curve_i = Curves.Quadratic_Bezier(*ControlPoints)
                self.Bezier_curves.append(curve_i)
            elif len(ControlPoints) == 4:
                curve_i = Curves.Cubic_Bezier(*ControlPoints)
                self.Bezier_curves.append(curve_i)

    def generate_points(self, total_points):
        t = np.linspace(0, self.total_curves, total_points)

        generated_points = []

        for i in range(self.total_curves):
            t_vals_curve_i = t[(t - i >= 0) & (t - i <= 1)] - i
            points_curve_i = self.Bezier_curves[i].generate_points_at_tvals(t_vals_curve_i)
            generated_points.append(points_curve_i)

        generated_points = np.vstack(generated_points)
        return generated_points


class CatmullRom_spline:
    
    def __init__(self, ControlPoints_list):
        self.CatmullRom_curves = []
        self.total_curves = len(ControlPoints_list) - 1


        a = ControlPoints_list[0]
        b = ControlPoints_list[1]
        GhostPoint_0 = (a[0] + (a[0] - b[0]), a[1] + (a[1] - b[1]))
        a = ControlPoints_list[-2]
        b = ControlPoints_list[-1]
        GhostPoint_1 = (b[0] + (b[0] - a[0]), b[1] + (b[1] - a[1]))
        ControlPoints_list.insert(0, GhostPoint_0)
        ControlPoints_list.append(GhostPoint_1)
        print(ControlPoints_list)

        for i in range(self.total_curves):
            curve_i = Curves.CatmullRom_curve(ControlPoints_list[i], ControlPoints_list[i + 1], ControlPoints_list[i + 2], ControlPoints_list[i + 3])
            self.CatmullRom_curves.append(curve_i)
        
    def generate_points(self, total_points):
        t = np.linspace(0, self.total_curves, total_points)

        generated_points = []

        for i in range(self.total_curves):
            t_vals_curve_i = t[(t - i >= 0) & (t - i <= 1)] - i
            points_curve_i = self.CatmullRom_curves[i].generate_points_at_tvals(t_vals_curve_i)
            generated_points.append(points_curve_i)

        generated_points = np.vstack(generated_points)
        return generated_points



class B_spline:

    def __init__(self, ControlPoints_list):
        self.Bspline_curves = []
        self.total_curves = len(ControlPoints_list)

        for ControlPoints in ControlPoints_list:
            curve_i = Curves.B_Spline_curve(*ControlPoints)
            self.Bspline_curves.append(curve_i)
    
    def generate_points(self, total_points):
        t = np.linspace(0, self.total_curves, total_points)

        generated_points = []

        for i in range(self.total_curves):
            t_vals_curve_i = t[(t - i >= 0) & (t - i <= 1)] - i
            points_curve_i = self.Bspline_curves[i].generate_points_at_tvals(t_vals_curve_i)
            generated_points.append(points_curve_i)

        generated_points = np.vstack(generated_points)
        return generated_points