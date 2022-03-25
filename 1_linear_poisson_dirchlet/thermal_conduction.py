#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
#######################################################################
### Solving linear Poisson Equation with direct solver              ###
#######################################################################
"""
import dolfin as dlfn
from dolfin import grad,dot
#======================================================================
mesh = dlfn.Mesh("step_domain.xml.gz")
# initial mesh refinement
for i in range(4):
    mesh = dlfn.refine(mesh)
space_dim = mesh.topology().dim()
#======================================================================
# function space
cell = mesh.ufl_cell()
elemTemp = dlfn.FiniteElement("Lagrange", cell, 1)
Vh = dlfn.FunctionSpace(mesh, elemTemp)
#======================================================================
# solution function
temp = dlfn.Function(Vh)
#======================================================================
# subdomains
gamma00 = dlfn.CompiledSubDomain("on_boundary")
#======================================================================
# subdomains markers
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim - 1)
facet_marker.set_all(0)
gamma00.mark(facet_marker, 1)
#======================================================================
# volume / surface element
dA = dlfn.Measure("ds", domain = mesh)
dV = dlfn.Measure("dx", domain = mesh)
#======================================================================
# test and trail function
delT = dlfn.TestFunction(Vh)
T = dlfn.TrialFunction(Vh)
# weak form
a = dot(grad(T), grad(delT)) * dV
l = dlfn.Constant(1.0) * delT * dV
#======================================================================
# Dirichlet boundary conditions
dirichlet_bcs = dlfn.DirichletBC(Vh, 0.0, facet_marker, 1)
#======================================================================
# problem formulation
problem = dlfn.LinearVariationalProblem(a, l, temp, bcs=dirichlet_bcs)
#======================================================================
# solver
problem_solver = dlfn.LinearVariationalSolver(problem)
problem_solver.parameters["linear_solver"] = "cg"
problem_solver.parameters["preconditioner"] = "hypre_amg"
problem_solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-12
problem_solver.parameters["krylov_solver"]["monitor_convergence"] = True
print problem_solver.parameters.str(True)
#======================================================================
# compute solution
problem_solver.solve()
#======================================================================
pvd_file = dlfn.File("solution_temp.pvd")
pvd_file << temp
#======================================================
dlfn.plot (temp)
