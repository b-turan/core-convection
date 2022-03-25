#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:48:03 2018

@author: Berkant / FEniCS tutorial
#######################################################################
### Solving nonlinear Poisson Equation with Picard iteration        ###
#######################################################################
"""
import dolfin as dlfn
from dolfin import grad, inner
#import matplotlib.pyplot as plt
#======================================================================
#New definitions
def q(u):
    return (dlfn.Constant(1)+u)**2                        # q(u) = 1 + u² 
#======================================================================
#Mesh definition & refinement
mesh = dlfn.Mesh("square_mod.xml")
for x in range(4):
    mesh = dlfn.refine(mesh)
space_dim = mesh.topology().dim()
ncell = mesh.num_cells()
#======================================================================
#Function space
cell = mesh.ufl_cell()        
# Notwendiger Befehl um die Zelle zu identifizieren
elemTemp = dlfn.FiniteElement("CG", cell, 1)    
# Bestimmt das Finite-Element, das den Zellen zugeordnet wird
Vh = dlfn.FunctionSpace(mesh, elemTemp) 
# Erstellt einen Funktionsraum, welcher diskret/endlich auf dem Mesh existiert 
# und beinhält nur Elemente des Typs "Continuous Galerkin"
# Alle Funktionen aus diesem Raum sind also endlich und linear
# (Index h weist auf die Diskretisierung des Funktionenraums V hin)
#======================================================================
#Subdomains / Unterbereiche für die Randbedingungen
gamma_left = dlfn.CompiledSubDomain("near(x[0], 0) && on_boundary")
gamma_right = dlfn.CompiledSubDomain("near(x[0], 1) && on_boundary")
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim -1)
facet_marker.set_all(0)
gamma_left.mark(facet_marker, 1)
gamma_right.mark(facet_marker, 2)
#======================================================================
# definition of volume / surface element (not explicitly neccessary)
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh, subdomain_data = facet_marker)
#======================================================================
#Trial- & Testfunction
u = dlfn.TrialFunction(Vh)      # "solution of current step"
v = dlfn.TestFunction(Vh)       # testfunction
#======================================================================
# Initial guess to linearize variational problem
start_temp = dlfn.Constant(0.)
u_k = dlfn.interpolate(start_temp, Vh)
#======================================================================
#linearized weak form /  Variational problem for picard iteration
a = q(u_k) * inner(grad(u), grad(v))*dV
L = dlfn.Constant(0.) * v * dV                 
#======================================================================
# Dirichlet boundary conditions
dbc_left = dlfn.DirichletBC(Vh, 2.0, facet_marker, 1)
dbc_right = dlfn.DirichletBC(Vh, 4.0, facet_marker, 2)
bcs = [dbc_left, dbc_right]
#======================================================================
# Picard iteration
u = dlfn.Function(Vh)   # neue Funktion, nach der gelöst wird 
eps = 1.0               # Startwert für epsillon 
tol = 1.0E-5            # tolerance
iter = 0                # Iterationsbeginn
max_iter = 50           # maximal Iterationen
problem = dlfn.LinearVariationalProblem(a, L, u, bcs)
problem_solver = dlfn.LinearVariationalSolver(problem)
import numpy as np
while abs(eps) > tol and iter < max_iter:
    iter += 1
    problem_solver.solve()
    diff = u.vector().array() - u_k.vector().array()
    eps = np.linalg.norm(diff, ord=np.Inf)
    u_k.assign(u)
    #plt.figure()
    #dlfn.plot(u, title ="temperatur")
    #plt.show()
    print "Norm:", eps
    
#======================================================================
#elemGradT = dlfn.VectorElement("DG", cell, 0)
#gradVh = dlfn.VectorFunctionSpace(mesh, "DG", 0)
#gradT = dlfn.project(grad(u), gradVh)
#======================================================================
pvd_file = dlfn.File("solution_temperatur.pvd")
pvd_file << u
#======================================================================
#pvd_file = dlfn.File("solution_gradient.pvd")
#pvd_file << gradT