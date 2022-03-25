#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:00:00 2018

@author: Berkant (+ FEniCS Tutorial)

#######################################################################
##           Solving nonlinear Poisson Equation                      ##
##           with NonLinearVariationalProblem/Solver                 ##
##                     (Newton method)                               ##
#######################################################################
"""
import dolfin as dlfn
from dolfin import inner, grad
#======================================================================
#New definitions
def q(u):
    return (1+u)**2                       # q(u) = 1 + u² 
#
def Dq(u):
    return 2*(1+u)                         # q'(u) = 2*(1+u)
#======================================================================
#Mesh definition & refinement
mesh = dlfn.Mesh("square_mod.xml")
for x in range(4):
    mesh = dlfn.refine(mesh)
space_dim = mesh.topology().dim()
ncell = mesh.num_cells()
dlfn.plot(mesh)
#======================================================================
#Element and FunctionSpace definition
cell = mesh.ufl_cell()        
elemTemp = dlfn.FiniteElement("CG", cell, 1)    
Vh = dlfn.FunctionSpace(mesh, elemTemp) 
#======================================================================
#Subdomains for boundaries
gamma_boundary1 = dlfn.CompiledSubDomain("near(x[0], 0.) && on_boundary")
gamma_boundary2 = dlfn.CompiledSubDomain("near(x[0], 1.) && on_boundary")
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim -1)
facet_marker.set_all(0)
gamma_boundary1.mark(facet_marker, 1)
gamma_boundary2.mark(facet_marker, 2)
#======================================================================
# definition of volume / surface element (not explicitly neccessary)
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh, subdomain_data = facet_marker)
#======================================================================
#Trial- & Testfunction
du = dlfn.TrialFunction(Vh)         
v = dlfn.TestFunction(Vh) 
u_ = dlfn.Function(Vh)             # most recently computed solution
#======================================================================
#Define nonlinear variational problem
F  = inner(q(u_)*grad(u_), grad(v))*dV
# manually derived (Gâteaux-Ableitung in Richtung "du")
J = inner(q(u_)*grad(du), grad(v))*dV + \
inner(Dq(u_)*du*grad(u_), grad(v))*dV     
#from dolfin import  derivative
#J = derivative(F, u_, du)
#======================================================================
#boundary conditions
dbc1 = dlfn.DirichletBC(Vh, 2., facet_marker, 1)
dbc2 = dlfn.DirichletBC(Vh, 4., facet_marker, 2)
bcs = [dbc1, dbc2]
#======================================================================
#problem
problem = dlfn.NonlinearVariationalProblem(F, u_, bcs, J)
solver = dlfn.NonlinearVariationalSolver(problem)
solver.solve()
#dlfn.solve(F == 0, u_, bcs, J=J)
#======================================================================
pvd_file = dlfn.File("solution_temperatur.pvd")
pvd_file << u_

