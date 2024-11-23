import matplotlib.pyplot as plt
import numpy as np

"""
Simulation of Rayleigh-Taylor Instability using the Finite Volume Method.
By Jack B, Sahat J, and Riddhi C.
"""
# To simulate the vertical section of two fluids and their interface
# we make a matrix of cells. We then add cells for simulation of the boundary
# conditions. To do this, we add "ghost cells". Reflective boundary
# is made at the top and the bottom, while periodic boundary at 
# the left and right.

def main():
    """
    Add docstring perchance
    """

    # Setting parameters
    N=64
    sizeX=0.5
    sizeY=1.5
    t=0
    tEnd=15

    # Creating the vertical section
    dx=sizeX/N
    vol_cell=dx**2
    xlin=np.linspace(0.5*dx, sizeX-0.5*dx,N)
    ylin=np.linspace(0.5*dx, sizeY-0.5*dx,3*N)


if __name__=="__main__":
    main()