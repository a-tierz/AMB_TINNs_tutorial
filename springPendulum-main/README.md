# Spring Pendulum simulator

## Overview
This module implements the equations of the elastic pendulum in Matlab, to generate the database
to train the different models in this TINNs tutorial. 

This work has been adapted by: 
*   PhD Pau Urdeitx - purdeitx@unizar.es --> [Web](https://amb.unizar.es/people/pau-urdeitx/)

The original code was written by: 
*   Siva Srinivas Kolukula - allwayzitzme@gmail.com --> [Web](http://sites.google.com/site/kolukulasivasrinivas/)

## How to execute:
This is a Matlab algorithm that generates a trajectory of the elastic pendulum. 
Files include:
* Main files:
    - SpringPendulum.m: This is the main file used to execute the model.
    - Spring Pendulum.pdf: here you will find the description of the model
    
* Additional files
    - Equation.m: Set the input values to pass onto ode45
    - Animation.m: To Animate the Spring Pendulum, the Equation is solved by MATLAB ode45, and running a loop over the time step; 
        for every time step, the position coordinates of the pendulum are updated. Phase plane plots of the spring and pendulum are plotted
    - arrowh.m: Draws a solid 2D arrowhead in the animation 
    - Spring.m: Calculates the position of a 2D spring 

* Output files:
    - data_rX.X_thetaYY_rev.mat: state variables of the elastic pendulum along the simulation (database to train) reversible
    - data_rX.X_thetaYY_dis.mat: state variables of the elastic pendulum along the simulation (database to train) dissipative

## Variables: 
|     Variable              |             Description                           |
|---------------------------| ------------------------------------------------- |
| `M`      	    | Mass of the pendulum             |
| `L`      	    | Natural length of the spring             |
| `K`      	    | Spring constant             |
| `d`      	    | Damping coefficient             |
| `duration`    | Duration of the simulation             |
| `fps`      	| Number of points to solve the ODE per second of sumulaiton   |
| `r`      	    | Initial position (radial): length of the spring             |
| `Phi`      	| Initial position (angular): angle of the spring             |
| `rdot`      	| Initial velocity: radial velocity            |
| `Phidot`      | Initial velocity: angular velocity           |

## Key Features
- Gravity: In case you need to, you can consider the effects of the gravity potential (grav = true/false)
- Dissipation: In case you need, you can dissipative dynamics by changing the damping coefficient:
    - d = 0 -> reversible
    - d > 0 -> dissipative
- Random initial positions: In case you need to, you can define different initial conditions to generate your database. 
- data.mat: Outputs the state variables, Z=(qx, qy, px, py), and time increment, dt, in Matlab format to train your model.


