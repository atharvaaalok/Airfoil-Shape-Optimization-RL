import numpy as np
import matplotlib.pyplot as plt
from Aerodynamics import Aerodynamics

import time

import concurrent.futures



def L_by_D_func(airfoil_name):
     # Get coordinates of airfoil
    airfoil_coordinates = np.loadtxt('Aerodynamics/Airfoil_Database/' + airfoil_name + '.dat')

    # Create airfoil object to analyze properties
    airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)

    # Get lift-to-drag ratio
    Reynolds_num = 1e6
    L_by_D_ratio = airfoil.get_L_by_D(Reynolds_num)
    return L_by_D_ratio


airfoil_name_list = ['NACA0006', 'NACA0009', 'NACA0012', 'NACA1408', 'NACA2412', 'NACA4412']


if __name__ == '__main__':

    # Time sequential compute
    print('Sequential Compute' + '\n' + '-' * 30)
    start = time.perf_counter()

    for airfoil_name in airfoil_name_list:
        L_by_D_ratio = L_by_D_func(airfoil_name)
        print(L_by_D_ratio)

    finish = time.perf_counter()
    print()
    print(f'Finished in {round(finish - start, 2)} second(s)')


    # Time parallel compute
    print('\n' + 'Parallel Compute' + '\n' + '-' * 30)
    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers = 60) as executor:
        results = executor.map(L_by_D_func, airfoil_name_list)
        # Map gives results in the order they were started

        for result in results:
            print(result)

    finish = time.perf_counter()
    print()
    print(f'Finished in {round(finish - start, 2)} second(s)')