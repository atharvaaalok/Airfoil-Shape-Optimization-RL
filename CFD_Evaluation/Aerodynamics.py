import subprocess
import numpy as np


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

    # Make airfoil coordinate data file
    airfoil_coord_filename = 'Airfoil_Coordinates.dat'
    np.savetxt(airfoil_coord_filename, airfoil.coordinates, delimiter = ',')

    # Start Xfoil
    xfoil = subprocess.Popen('xfoil.exe', stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True)

    # Set CFD evaluation parameters
    panel_count = 450
    LE_TE_panel_density_ratio = 1
    Reynolds_num = Re
    max_iter_count = 100

    # Define the sequence of commands to execute to calculate the aerodynamic properties
    command_list = ['PLOP\n',                               # Go to plotting options menu
                    'G\n',                                  # Switch off graphical display 
                    '\n',
                    f'load {airfoil_coord_filename}\n',
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
    # try:
    #     xfoil_stdout, xfoil_stderr = xfoil.communicate(timeout = 0.8)
    # except:
    #     aerodynamic_properties = None
    #     return aerodynamic_properties

    # Extract the aerodynamic properties
    coefficients = xfoil_stdout.splitlines()[-4].split()

    # Check if the values are numbers, if not, then it did not converge and return None
    try:
        aerodynamic_properties = {'CL': float(coefficients[1]), 'CD': float(coefficients[2])}
    except:
        aerodynamic_properties = None

    return aerodynamic_properties