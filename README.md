# core-convenction
This repository provides several implementations of the Finite-Element-Method (FEM) for solving common problems in continuum mechanics. The underlying partial differential equations (PDEs) are solved utilizing the popular open-source platform [FEniCS](https://fenicsproject.org/). All codes were written as part of my bachelor thesis "Numerical Modeling and Simulation of Convection in Earth's Outer Core via Finite Element Method". 

The programs range from simple introductory problems such as the cavity flow to more advanced problems like the 2-dimensional mantle/core convection. The latter was realized by linking the Navier-Stokes-Equation with the Heat Equation using the Boussinesq-approximation. Furthermore, the derived PDEs were discretized with advanced numerical approaches, e.g. the [implicit-explicit-scheme](https://www.sciencedirect.com/science/article/pii/S0168927497000561) (IMEX, 2nd order), which is a multistep time-discretization scheme. Performances with respect to accuracy were analyzed on benchmark problems.

## Remark
The code was not intended for the public and was therefore not implemented according to specific coding standards. 
