#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import dolfin as dlfn
import mshr
from dolfin import inner, grad, div, dot
import numpy as np
import matplotlib.pyplot as plt
#==============================================================================
dlfn.parameters["form_compiler"]["cpp_optimize"] = True
comm = dlfn.mpi_comm_world()
mpi_comm = dlfn.mpi_comm_world()
mpi_rank = dlfn.MPI.rank(mpi_comm)
print mpi_rank
print "The current version of FEniCS is ", dlfn.dolfin_version()
#==============================================================================
# time configuration
t = 0.
t_end = 250.
delta_t = 1e-2
dt = dlfn.Constant(delta_t)
cnt = 0 # iteration counter
cfl = 1. # cfl-number
#==============================================================================
# dimensionless constants
rayleigh = 3.4e5
prandtl = 0.71
#==============================================================================
# ufl constants
Ra = dlfn.Constant(rayleigh)
Pr = dlfn.Constant(prandtl)
#==============================================================================
# mesh creation
r_outer = 1. # outer radius
r_inner = r_outer/2. # inner radius
center = dlfn.Point(0., 0.)
domain = mshr.Circle(center, r_outer) \
       - mshr.Circle(center, r_inner)
mesh = mshr.generate_mesh(domain, 25)
space_dim = mesh.topology().dim()
n_cells = mesh.num_cells()
#==============================================================================
# subdomains for boundaries
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim-1)
facet_marker.set_all(0)
hmin = mesh.hmin() # abmessung kleinstes Element
# inner circle boundary
class InnerCircle(dlfn.SubDomain):
    def inside(self, x, on_boundary):
        tol = hmin/2. # tolerance: half length of smallest element
        result = abs(dlfn.sqrt(x[0]**2 + x[1]**2) - r_inner) < tol
        return result
# outer cirlce boundary
class OuterCircle(dlfn.SubDomain):
    def inside(self, x, on_boundary):
        tol = hmin/2. # tolerance: half length of smallest element
        result = abs(dlfn.sqrt(x[0]**2 + x[1]**2) - r_outer) < tol
        return result
# mark boundaries
gamma_inner = InnerCircle()
gamma_inner.mark(facet_marker, 1)
gamma_outer = OuterCircle()
gamma_outer.mark(facet_marker, 2)
# save pvd file
dlfn.File('subdomains.pvd') << facet_marker
#==============================================================================
# element and function space definition
cell = mesh.ufl_cell()
# taylor-hood element
elemV = dlfn.VectorElement("CG", cell, 2)
elemP = dlfn.FiniteElement("CG", cell, 1)
elemT = dlfn.FiniteElement("CG", cell, 1) 
mixedElem = dlfn.MixedElement([elemV, elemP, elemT])
Wh = dlfn.FunctionSpace(mesh, mixedElem)
ndofs = Wh.dim()
ndofs_velocity = Wh.sub(0).dim()
ndofs_pressure = Wh.sub(1).dim()
ndofs_temperature = Wh.sub(2).dim()
print "DOFs velocity : ", ndofs_velocity, "DOFs pressure : ", ndofs_pressure, \
      "DOFs temperature : ", ndofs_temperature
vortex_space = dlfn.FunctionSpace(mesh, "DG", 1)
radial_vector_space = dlfn.FunctionSpace(mesh, elemV)
#==============================================================================
# boundary conditions
null_vector = dlfn.Constant((0.0, 0.0))
temp_hot = dlfn.Expression("0.5 * t / 2.", degree= 2, t= 0)
temp_cold = dlfn.Expression("-0.5 * t / 2.", degree= 2, t= 0)
temp_inner = dlfn.Constant(0.5)
temp_outer = dlfn.Constant(-0.5)
# initialize empty list
bcs = []
# no slip bc on all boundaries
bcs.append(dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, 1))
bcs.append(dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, 2))
# temperature bcs on left and right boundary
bcs.append(dlfn.DirichletBC(Wh.sub(2), temp_inner, facet_marker, 1))
bcs.append(dlfn.DirichletBC(Wh.sub(2), temp_outer, facet_marker, 2))
#==============================================================================
# definition of volume / surface element
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh)
A = dlfn.assemble(1.*dV) # "volume" or rather area of geometry
#n = dlfn.FacetNormal(mesh)
#==============================================================================
# trial and test function
(del_v, del_p, del_T) = dlfn.TestFunctions(Wh)
(dv, dp, dT) = dlfn.TrialFunctions(Wh)
#==============================================================================
# solution functions
sol = dlfn.Function(Wh)
sol0 = dlfn.Function(Wh)
v, p, T = dlfn.split(sol)
v0, p0, T0 = dlfn.split(sol0)
#==============================================================================
# define auxiliary operators
def a_operator(phi, psi):
    return inner(grad(phi), grad(psi)) * dV
def b_operator(phi, psi):
    return div(phi) * psi * dV
def c_operator(phi, chi, psi):
    return  dot(dot(grad(chi), phi), psi) * dV
#==============================================================================
# ufl-constants for SBDF scheme
one = dlfn.Constant(1.)
two = dlfn.Constant(2.)
half = dlfn.Constant(1./2.)
three_half = dlfn.Constant(3./2.)
# unit vector in radial direction
radial_unit = dlfn.Expression(('1./sqrt(pow(x[0],2)+pow(x[1],2)) * x[0]', 
                       '1./sqrt(pow(x[0],2)+pow(x[1],2)) * x[1]'), degree=2)
e_r = dlfn.interpolate(radial_unit, radial_vector_space)
#==============================================================================
pvd_velocity = \
    dlfn.File("Euler_{0:1.1e}_{1:1.1e}_{2}s/solution_velocity.pvd".\
                         format(int(rayleigh), float(delta_t), int(t_end)))
pvd_pressure = \
    dlfn.File("Euler_{0:1.1e}_{1:1.1e}_{2}s/solution_pressure.pvd".\
                         format(int(rayleigh), float(delta_t), int(t_end)))
pvd_temperature = \
    dlfn.File("Euler_{0:1.1e}_{1:1.1e}_{2}s/solution_temperature.pvd".\
                         format(int(rayleigh), float(delta_t), int(t_end)))
#==============================================================================
# define nonlinear variational problem (CN scheme)
# momentum equation          
F_v = (inner(v, del_v) - inner(v0, del_v)) * dV \
    + dt * dlfn.sqrt(Pr/Ra) * a_operator(v, del_v) \
    + dt * c_operator(v, v, del_v) \
    - dt * b_operator(v, del_p) \
    - dt * b_operator(del_v, p) \
    - dt * T * dot(e_r, del_v) * dV
# energy equation
F_T = (dot(T, del_T) - dot(T0, del_T)) * dV \
    + dt / dlfn.sqrt(Ra*Pr) * a_operator(T, del_T) \
    + dt * c_operator(v, T, del_T)
F_sum = F_v + F_T
J_newton = dlfn.derivative(F_sum, sol)
problem = dlfn.NonlinearVariationalProblem(F_sum, sol, bcs=bcs, J=J_newton)
solver = dlfn.NonlinearVariationalSolver(problem) 
#==============================================================================
# time loop
dlfn.tic()
while t < t_end: #and cnt <= n_steps:
    # solve nonlinear problem
    solver.solve()
    v, p, T = sol.split(deepcopy=True)
    # write output
    if cnt%2 == 0:
        pvd_velocity << (v, t)
        pvd_pressure << (p, t)
        pvd_temperature << (T, t)
        if mpi_rank == 0:
            print "t = {0:6.4f}".format(t)
            #print "Iteration: ", cnt
    if cnt%15 == 0:
        kin_e = dlfn.assemble(dlfn.Constant(1./(2.*A)) * dot(v, v) * dV)
        if kin_e > 1e12:
            raise ValueError("Instability occured!")
    # update for next iteration
    sol0.assign(sol)
    t += delta_t
    cnt += 1
if mpi_rank == 0:
    print 'elapsed simulation time in seconds: ', dlfn.toc()
    # cfl-configuration
#    v_nodal_values = v.vector().array()
#    vmax = np.max(np.abs(v_nodal_values))
#    delta_t = cfl*hmin/vmax
#    dt = dlfn.Constant(delta_t)
#    if t < 2:
#        temp_hot.t += delta_t
#        temp_cold.t += delta_t