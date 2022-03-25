# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:44:24 2018

@author: Berkant Turan
"""
import dolfin as dlfn 
from dolfin import grad,dot
#======================================================================
mesh = dlfn.Mesh("lshape.xml")
#initial mesh refinement
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
# subdomains & markers
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim - 1)
facet_marker.set_all(0)
gamma01 = dlfn.CompiledSubDomain('near(x[0], 0.) && on_boundary')
gamma02 = dlfn.CompiledSubDomain('near(x[1], 1.) && on_boundary')
gamma_neumann_left = dlfn.CompiledSubDomain('near(x[0], -1.) && on_boundary')
gamma_neumann_bottom = dlfn.CompiledSubDomain('near(x[1], -1.)&& on_boundary ')
gamma01.mark(facet_marker, 1)
gamma02.mark(facet_marker, 2)
gamma_neumann_left.mark(facet_marker, 3)
gamma_neumann_bottom.mark(facet_marker, 4)
#======================================================================
# volume / surface element
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh, subdomain_data = facet_marker)
#======================================================================
# test and trail function
delT = dlfn.TestFunction(Vh)
T = dlfn.TrialFunction(Vh)
# weak form with Neumann-Condition
left_neumann = dlfn.Constant(40.)
bottom_neumann = dlfn.Constant(20.)
a = dot(grad(T), grad(delT)) * dV
l = dlfn.Constant(100.0) * delT * dV  - left_neumann * delT *dA(3) \
 - bottom_neumann * delT *dA(4)
#======================================================================
# define dirichlet boundary conditions
dirichlet_bc_x0 = dlfn.DirichletBC(Vh, 2.0, facet_marker, 1)
dirichlet_bc_y1 = dlfn.DirichletBC(Vh, -30.0, facet_marker, 2)
bcs = [dirichlet_bc_x0, dirichlet_bc_y1]
#======================================================================
# problem formulation
problem = dlfn.LinearVariationalProblem(a, l, temp, bcs)
#======================================================================
# solver
problem_solver = dlfn.LinearVariationalSolver(problem)
problem_solver.parameters["linear_solver"] = "cg"
problem_solver.parameters["preconditioner"] = "hypre_amg"
problem_solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-12
problem_solver.parameters["krylov_solver"]["monitor_convergence"] = True
print(problem_solver.parameters.str(True))
#======================================================================
# compute solution
problem_solver.solve()
#======================================================================
#elemGradT = dlfn.VectorElement("DG", cell, 0)
gradVh = dlfn.VectorFunctionSpace(mesh, "DG", 0)
gradT = dlfn.project(grad(temp), gradVh)
#======================================================================
pvd_file = dlfn.File("solution_temp.pvd")
pvd_file << temp
#======================================================
pvd_file = dlfn.File("solution_grad.pvd")
pvd_file << gradT
