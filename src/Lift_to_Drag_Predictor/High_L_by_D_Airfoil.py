import numpy as np

from ..CFD.Aerodynamics import Aerodynamics


# airfoil_coordinates = np.array([[1, 0], [0.75, 0.05], [0.5, 0.10], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.10], [0.75, -0.05], [1, 0]])
airfoil_coordinates = np.array([[1, 0], [0.75, 0.05], [0.625, 0.08], [0.5, 0.1], [0.25, 0.10], [0, 0], [0.25, -0.004], [0.5, 0.005], [0.625, 0.008], [0.75, 0.015], [1, 0]])
# Visualize the airfoil in xfoil
airfoil_name = 'my_airfoil'
airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)
airfoil.visualize()

Reynolds_num = 1e6
print(airfoil.get_L_by_D(Reynolds_num))