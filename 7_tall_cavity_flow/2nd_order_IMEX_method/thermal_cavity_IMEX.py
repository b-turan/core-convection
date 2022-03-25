#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import dolfin as dlfn
from dolfin import inner, grad, div, dot
import numpy as np
import matplotlib.pyplot as plt
dlfn.parameters["form_compiler"]["cpp_optimize"] = True
comm = dlfn.mpi_comm_world()
mpi_comm = dlfn.mpi_comm_world()
mpi_rank = dlfn.MPI.rank(mpi_comm)
print mpi_rank
#==============================================================================
postprocessor = False # post processing True/False 
#==============================================================================
# definition of auxillary functions
# define function for calculation of nusselt number
def nusselt_fun(T, wall):
    # TO DO: get number of facet_marker automatically
    H = dlfn.assemble(1. * dA(1)) 
    nusselt = dlfn.assemble(abs(T.dx(0)) * dA(wall)) / H
    return nusselt
# define function for calculation of vorticity 
def vorticity_fun(v):
    omega = v[1].dx(0) - v[0].dx(1)
    omega_metric = dlfn.sqrt(dlfn.assemble(dlfn.Constant(1./(2.*A)) \
                                           * omega**2 * dV))
    return omega_metric
# append values to lists for post processor
def append_fun(v, p, T):
    # time
    t_list.append(t)
    # save values at time history point 1
    vx_list1.append(v(np.array(point1))[0])
    vy_list1.append(v(np.array(point1))[1])
    T_list1.append(T(np.array(point1)))
    p_list1.append(p(np.array(point1)))
    # pressure differences
    p_diff_list14.append(p(np.array(point1)) - p(np.array(point4)))
    p_diff_list51.append(p(np.array(point5)) - p(np.array(point1)))
    p_diff_list35.append(p(np.array(point3)) - p(np.array(point5)))
    # average velocity metric
    v_metric_list.append(dlfn.sqrt(kin_e))
    # skewness metric 
    skew_metric_list12.append(T(np.array(point1)) + T(np.array(point2)))
    skew_metric_vx_list12.append(v(np.array(point1))[0]
                                + v(np.array(point2))[0])
    # average vorticity metric
    omega_metric_list.append(omega_metric)
    # nusselt lists
    nusselt_left_list.append(nusselt_left)
    nusselt_right_list.append(nusselt_right)
    # list of iteration
    iteration_list.append(cnt)
    return 
#==============================================================================
# dimensionless constants
rayleigh = 3.4e5
prandtl = 0.71
#==============================================================================
# ufl constants
Ra = dlfn.Constant(rayleigh)
Pr = dlfn.Constant(prandtl)
#==============================================================================
# initial time step
delta_t = 1e-3
dt = dlfn.Constant(delta_t)
#==============================================================================
t = 0.
t_end = 1.
cnt = 0
#==============================================================================
# mesh definition and refinement
mesh = dlfn.RectangleMesh(dlfn.Point(0, 0),dlfn.Point(1,8),25,100,"right")
space_dim = mesh.topology().dim()
n_cells = mesh.num_cells()
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
#==============================================================================
# subdomains for boundaries
gamma_left = dlfn.CompiledSubDomain("near(x[0], 0.) && on_boundary")
gamma_right = dlfn.CompiledSubDomain("near(x[0], 1.) && on_boundary")
gamma_top = dlfn.CompiledSubDomain("near(x[1], 8.) && on_boundary")
gamma_bottom = dlfn.CompiledSubDomain("near(x[1], 0.) && on_boundary")
# marking subdomains
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim -1)
facet_marker.set_all(0)
gamma_left.mark(facet_marker, 1)
gamma_right.mark(facet_marker, 2)
gamma_top.mark(facet_marker, 3)
gamma_bottom.mark(facet_marker, 4)
dlfn.File('subdomains.pvd') << facet_marker
raise ValueError()
#==============================================================================
# boundary conditions
null_vector = dlfn.Constant((0.0, 0.0))
temp_left = dlfn.Constant(0.5)
temp_right = dlfn.Constant(-0.5)
# initialize empty list
bcs = []
# no slip bc on all boundaries
bcs.append(dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, 1))
bcs.append(dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, 2))
bcs.append(dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, 3))
bcs.append(dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, 4))
# temperature bcs on left and right boundary
bcs.append(dlfn.DirichletBC(Wh.sub(2), temp_left, facet_marker, 1))
bcs.append(dlfn.DirichletBC(Wh.sub(2), temp_right, facet_marker, 2))
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
sol00 = dlfn.Function(Wh)
v0, p0, T0 = dlfn.split(sol0)
v00, p00, T00 = dlfn.split(sol00)
#==============================================================================
# define auxiliary operators
def a_operator(phi, psi):
    return inner(grad(phi), grad(psi)) * dV
def b_operator(phi, psi):
    return div(phi) * psi * dV
def c_operator(phi, chi, psi):
    return  dot(dot(grad(chi), phi), psi) * dV
#==============================================================================
# ufl constants for SBDF scheme
one = dlfn.Constant(1.)
two = dlfn.Constant(2.)
half = dlfn.Constant(1./2.)
three_half = dlfn.Constant(3./2.)
# unit vector in direction of y
j = dlfn.Constant((0.0, 1.)) 
#==============================================================================
# define linear variational problem
# momentum equation
lhs_momentum = three_half * dot(dv, del_v) *  dV \
             + dt * dlfn.sqrt(Pr/Ra) * a_operator(dv, del_v) \
             + dt * two * c_operator(v0, dv, del_v) \
             - dt * c_operator(v00, dv, del_v) \
             - dt * b_operator(del_v, dp) \
             - dt * b_operator(dv, del_p) \
             - dt * dT * dot(j, del_v) * dV
rhs_momentum = two * dot(v0, del_v) * dV \
             - half * dot(v00, del_v) * dV
# energy equation
lhs_energy = three_half * dot(dT, del_T) * dV \
           + dt / dlfn.sqrt(Ra*Pr) * a_operator(dT, del_T)
rhs_energy = two * T0 * del_T * dV \
           - half * T00 * del_T * dV \
           - dt * two * c_operator(v0, T0, del_T) \
           + dt * c_operator(v00, T00, del_T)
lhs = lhs_momentum + lhs_energy
# full problem
lhs = lhs_momentum + lhs_energy
rhs = rhs_momentum + rhs_energy
problem = dlfn.LinearVariationalProblem(lhs, rhs, sol, bcs=bcs)
solver = dlfn.LinearVariationalSolver(problem)
#==============================================================================
pvd_velocity = \
    dlfn.File("pvd_{0:1.1e}_{1:1.1e}_{2}s/solution_velocity.pvd".\
                         format(int(rayleigh), float(delta_t), int(t_end)))
pvd_pressure = \
    dlfn.File("pvd_{0:1.1e}_{1:1.1e}_{2}s/solution_pressure.pvd".\
                         format(int(rayleigh), float(delta_t), int(t_end)))
pvd_temperature = \
    dlfn.File("pvd_{0:1.1e}_{1:1.1e}_{2}s/solution_temperature.pvd".\
                         format(int(rayleigh), float(delta_t), int(t_end)))
#==============================================================================
# time history points (post processing) 
point1 = [0.1810, 7.3700]  # [x-coordinate, y-coordinate]
point2 = [0.8190, 0.6300]
point3 = [0.1810, 0.6300]
point4 = [0.8190, 7.3700]
point5 = [0.1810, 4.0000]
#==============================================================================
# initialize lists (post processing)
t_list, iteration_list = [], []
T_list1, vx_list1, vy_list1, p_list1 = [], [] , [], []
v_metric_list, skew_metric_list12, omega_metric_list = [], [], []
skew_metric_vx_list12 = []
p_diff_list14, p_diff_list51, p_diff_list35 = [], [], []
nusselt_left_list, nusselt_right_list= [], []
#==============================================================================
# time loop
dlfn.tic()
# n_steps = 10000
while t < t_end: #and cnt <= n_steps:
    # solve linear problem
    solver.solve()
    v, p, T = sol.split()
    # write output
    if cnt%10 == 0:
        pvd_velocity << (v, t)
        pvd_pressure << (p, t)
        pvd_temperature << (T, t)
        if mpi_rank == 0:
            print "t = {0:6.4f}".format(t)
            print "Iteration: ", cnt
    if cnt%1 == 0:
        kin_e = dlfn.assemble(dlfn.Constant(1./(2.*A)) * dot(v, v) * dV)
        if kin_e > 1e12:
            raise ValueError("Instability occured!")
        # save values for post processor
        omega_metric = vorticity_fun(v) # average vorticity metric
        nusselt_left = nusselt_fun(T=T, wall=1) # left wall nusselt number
        nusselt_right = nusselt_fun(T=T, wall=2) # right wall nusselt number
        append_fun(v, p, T)  # append new values to lists
    if cnt%10 == 0:
        header = "0. Iteration  " + "1. Time  " + "2. Temperature(1)  " +  \
         "3. v_x(1)  " + "4. v_y(1)  " + "5. average velocity metric  " + \
         "6. skewness metric T_eps(12)  " + "7. pressure diff(1-4)  " + \
         "8. pressure diff(5-1)  " + "9. pressure diff(3-5)  " + \
         "10. vorticity metric  " + \
         "11. skewness metric vx_eps(12)" + "12.nusselt_left  " + \
         "13. nusselt_right  " + \
         "delta_t = {0:1.1e}".format(float(delta_t))
        np.savetxt('values_inbetween_ray_{0:1.1e}_dt_{1:1.1e}_{2}s.gz'.\
                    format(int(rayleigh), float(delta_t), int(t_end)), \
          np.column_stack((iteration_list, t_list, T_list1, vx_list1, 
                           vy_list1, v_metric_list, skew_metric_list12, \
                           p_diff_list14, p_diff_list51, p_diff_list35, \
                           omega_metric_list, skew_metric_vx_list12, 
                           nusselt_left_list)), 
                           fmt= '%1.6e', header = header)
    # update for next iteration
    sol00.assign(sol0)
    sol0.assign(sol)
    t += delta_t
    cnt += 1
if mpi_rank == 0:
    print 'elapsed simulation time in seconds: ', dlfn.toc()
#==============================================================================
#==========================  post processor  ==================================
#==========================  TO DO: finish  ===================================
#==============================================================================
# save data
header = "0. Iteration  " + "1. Time  " + "2. Temperature(1)  " +  \
         "3. v_x(1)  " + "4. v_y(1)  " + "5. average velocity metric  " + \
         "6. skewness metric T_eps(12)  " + "7. pressure diff(1-4)  " + \
         "8. pressure diff(5-1)  " + "9. pressure diff(3-5)  " + \
         "10. vorticity metric  " + \
         "11. skewness metric vx_eps(12)  " + "nusselt left  " + \
         "nusselt right  " + \
         "delta_t = {0:1.1e}".format(float(delta_t))
np.savetxt('saved_values/ray_{0:1.1e}_dt_{1:1.1e}_{2}s.gz'.format(int(rayleigh),\
           float(delta_t), int(t)), \
          np.column_stack((iteration_list, t_list, T_list1, vx_list1, 
                           vy_list1, v_metric_list, skew_metric_list12,
                           p_diff_list14, p_diff_list51, p_diff_list35,
                           omega_metric_list, skew_metric_vx_list12, 
                           nusselt_left_list, nusselt_right_list)), 
                           fmt= '%1.6e', header = header)
#==============================================================================