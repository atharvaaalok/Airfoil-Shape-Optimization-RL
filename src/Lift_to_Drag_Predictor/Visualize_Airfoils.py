import numpy as np
import matplotlib.pyplot as plt

from ..CFD.Aerodynamics import Aerodynamics

import os

# File paths
directory_path = os.path.dirname(__file__)
numpy_arr_file_path = directory_path + '/Dataset/Arrays_as_rows_a_scaling_100.txt'
rewards_file_path = directory_path + '/Dataset/Rewards_as_rows_a_scaling_100.txt'


# Function to read a random line from the file and convert it into a NumPy vector
def read_random_line(file_path):
    with open(file_path, 'r') as file:
        # Count the total number of lines in the file
        num_lines = sum(1 for line in file)
    
    # Generate a random line number within the range of total lines
    random_line_number = np.random.randint(0, num_lines - 1)
    
    # Read the selected random line from the file
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file):
            if line_num == random_line_number:
                # Convert the line into a NumPy vector
                numpy_vector = np.fromstring(line, dtype = float, sep=' ')
                return numpy_vector  # Return the NumPy vector


plt.ion()
plt.xlabel('x')
plt.ylabel('y')
plt.title('My Airfoil')
plt.grid(True)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

total_airfoils_visualize = 100

for i in range(total_airfoils_visualize):
    airfoil_coordinates = read_random_line(numpy_arr_file_path).reshape(-1, 2)
    # Visualize the airfoil in xfoil
    airfoil_name = 'my_airfoil'
    airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)
    # airfoil.visualize()
    plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1], marker = 'o')
    plt.pause(0.1)

    Reynolds_num = 1e6
    print(airfoil.get_L_by_D(Reynolds_num))


plt.ioff()