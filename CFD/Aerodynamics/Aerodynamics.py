import subprocess
import numpy as np
import os

class Airfoil:
    def __init__(self, airfoil_coordinates, airfoil_name):
        self.coordinates = np.array(airfoil_coordinates)
        self.name = airfoil_name
    
    def get_aerodynamic_properties(self, Re, angle_of_attack = 0):
        aerodynamic_properties = CFD(self, Re, angle_of_attack)
        return aerodynamic_properties

    def get_L_by_D(self, Re, angle_of_attack = 0):
        aerodynamic_properties = self.get_aerodynamic_properties(Re, angle_of_attack)
        if aerodynamic_properties is None:
            return None
        return aerodynamic_properties['CL'] / aerodynamic_properties['CD']

    def get_CL(self, Re, angle_of_attack = 0):
        aerodynamic_properties = self.get_aerodynamic_properties(Re, angle_of_attack)
        if aerodynamic_properties is None:
            return None
        return aerodynamic_properties['CL']

    def get_CD(self, Re, angle_of_attack = 0):
        aerodynamic_properties = self.get_aerodynamic_properties(Re, angle_of_attack)
        if aerodynamic_properties is None:
            return None
        return aerodynamic_properties['CD']


def CFD(airfoil, Re, angle_of_attack = 0):
    
    coordinate_file_directory = '/Airfoil_Coordinates/'

    # Get relative path to the directory where to store the airfoil coordinate files
    top_level_script_directory = os.getcwd()
    aerodynamics_module_directory = os.path.dirname(__file__)
    relative_path_xfoil = aerodynamics_module_directory[len(top_level_script_directory) + 1:] + coordinate_file_directory

    # Get the different file paths necessary to save the coordinate file and then retrieve using airfoil
    airfoil_coord_filename = airfoil.name + '.dat'
    airfoil_save_path = os.path.dirname(__file__) + coordinate_file_directory + airfoil_coord_filename
    xfoil_file_path = relative_path_xfoil + airfoil_coord_filename

    # Save the airfoil coordinate file
    np.savetxt(airfoil_save_path, airfoil.coordinates, delimiter = ',')

    # Start Xfoil
    xfoil_path = os.path.dirname(__file__) + '/xfoil.exe'
    xfoil = subprocess.Popen(xfoil_path, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)

    # Set CFD evaluation parameters
    panel_count = 120
    LE_TE_panel_density_ratio = 1
    Reynolds_num = Re
    max_iter_count = 100

    # Define the sequence of commands to execute to calculate the aerodynamic properties
    command_list = ['PLOP\n',                               # Go to plotting options menu
                    'G\n',                                  # Switch off graphical display 
                    '\n',
                    f'load {xfoil_file_path}\n',
                    f'{airfoil.name}\n',
                    'PPAR\n',                               # go to panel menu
                    f'n {panel_count}\n',                   # set panel count
                    f't {LE_TE_panel_density_ratio}\n',     # set Leading Edge to Trailing Edge panel density
                    '\n',
                    '\n',
                    'OPER\n',                               # Go to operations menu to run the simulation
                    'visc\n'
                    f'{Reynolds_num}\n'
                    f'iter {max_iter_count}\n'
                    'PACC\n'                                # Turn on polar accumulation
                    '\n',
                    '\n',
                    f'alfa {angle_of_attack}\n',
                    'PLIS\n',                               # List the polar values
                    '\n',
                    'QUIT\n'
                    ]
    
    # Execute the commands and close Xfoil
    xfoil.stdin.write(''.join(command_list))
    xfoil.stdin.flush()
    xfoil.stdin.close()
    
    # Get the outputs from xfoil
    xfoil_stdout, xfoil_stderr = xfoil.communicate()

    # Remove the extra airfoil coordinate file created
    os.remove(airfoil_save_path)

    # Extract the aerodynamic properties
    coefficients = xfoil_stdout.splitlines()[-4].split()

    # Check if the values are numbers, if not, then it did not converge and return None
    try:
        aerodynamic_properties = {'CL': float(coefficients[1]), 'CD': float(coefficients[2])}
    except:
        aerodynamic_properties = None

    return aerodynamic_properties