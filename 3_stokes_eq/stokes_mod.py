# -*- coding: utf-8 -*-
import dolfin as dlfn
from dolfin import CompiledSubDomain, Measure, Mesh, refine, FunctionSpace,\
                   FiniteElement, VectorElement, MeshFunction, TrialFunctions,\
                   TestFunctions, DirichletBC, Function
from dolfin import grad, inner, div, Constant
from scipy.sparse import csr_matrix
#======================================================================
#mesh + refinement
mesh = Mesh("square_mod.xml")
for i in range(2):
   mesh = refine(mesh)
space_dim = mesh.topology().dim()
elementcount = mesh.num_cells()
dlfn.plot(mesh)
#======================================================================
# function space
cell = mesh.ufl_cell()
P1 = FiniteElement("CG", cell, 1)
P2 = VectorElement("CG", cell, 2)
mixed_space = P2 * P1
W = FunctionSpace(mesh, mixed_space)
ndofs = W.dim()
ndofs_velocity = W.sub(0).dim()
ndofs_pressure = W.sub(1).dim()
#======================================================================
# dof indices
dof_indices_vel = W.sub(0).dofmap().dofs()
dof_indices_p = W.sub(1).dofmap().dofs()
#======================================================================
# subdomain 
gamma_left = CompiledSubDomain("near(x[0], 0.) && on_boundary")
gamma_bottom = CompiledSubDomain("near(x[1], 0.) && on_boundary")
gamma_right = CompiledSubDomain("near(x[0], 1.) && on_boundary")
gamma_top = CompiledSubDomain("near(x[1], 1.) && on_boundary")
#======================================================================
# subdomain markers
facet_marker = MeshFunction("size_t", mesh, space_dim -1)
facet_marker.set_all(0)
gamma_left.mark(facet_marker, 1)
gamma_bottom.mark(facet_marker, 1)
gamma_right.mark(facet_marker, 1)
gamma_top.mark(facet_marker, 2)
#======================================================================
# volume / surface elements
dA = Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = Measure("dx", domain = mesh, subdomain_data = facet_marker)
#======================================================================
# Test- & TrialFunction
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
#======================================================================
#weak formulation
f = Constant((0.0, 0.0))
a = inner(grad(u), grad(v))*dV - div(v)*p*dV - q*div(u)*dV
l = inner(f, v)*dV
#======================================================================
A = csr_matrix(dlfn.assemble(a).array())
import matplotlib.pyplot as plt
plt.figure()
plt.spy(A)
B = csr_matrix((ndofs, ndofs))
for row in range(ndofs_velocity):
    for col in range(ndofs_velocity):
        if A[dof_indices_vel[row],dof_indices_vel[col]] != 0.0:
            B[row,col] = A[dof_indices_vel[row],dof_indices_vel[col]]
    for pcol in range(ndofs_pressure):
        col = ndofs_velocity + pcol
        if A[dof_indices_vel[row],dof_indices_p[pcol]] != 0.0:
            B[row,col] = A[dof_indices_vel[row],dof_indices_p[pcol]]
for prow in range(ndofs_pressure):
    row = ndofs_velocity + prow
    for col in range(ndofs_velocity):
        if A[dof_indices_p[prow],dof_indices_vel[col]] != 0.0:
            B[row,col] = A[dof_indices_p[prow],dof_indices_vel[col]]
    for pcol in range(ndofs_pressure):
        col = ndofs_velocity + pcol
        if A[dof_indices_p[prow],dof_indices_p[pcol]] != 0.0:
            B[row,col] = A[dof_indices_p[prow],dof_indices_p[pcol]]
plt.figure()
plt.spy(B)
#======================================================================
#dirichlet boundary
noslip = Constant((0.0, 0.0))
slip = Constant((5.0, 0.0))
bc_noslip = DirichletBC(W.sub(0), noslip, facet_marker, 1)
bc_slip = DirichletBC(W.sub(0), slip, facet_marker, 2)
bcs = [bc_noslip, bc_slip]
#======================================================================
Sol = Function(W)
#======================================================================
# problem formulation
problem = dlfn.LinearVariationalProblem(a, l, Sol, bcs)
#======================================================================
# solver
problem_solver = dlfn.LinearVariationalSolver(problem)
problem_solver.parameters["linear_solver"] = "direct"
#problem_solver.parameters["preconditioner"] = "hypre_amg"
#problem_solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-12
#problem_solver.parameters["krylov_solver"]["monitor_convergence"] = True
#print(problem_solver.parameters.str(True))
#======================================================================
# compute solution
problem_solver.solve()
(sol_u, sol_p) = Sol.split()
#======================================================================
pvd_file = dlfn.File("solution_velocity.pvd")
pvd_file << sol_u
#======================================================
pvd_file = dlfn.File("solution_pressure.pvd")
pvd_file << sol_p