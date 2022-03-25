#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Berkant 
#######################################################################
##           Solving unsteady Navier-Stokes-Equation                 ##
##           with NonLinearVariationalProblem/Solver                 ##
##          (implicit Euler - Karman Vortex Street)                  ##
#######################################################################
                    for more information see 
 http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html
"""
import dolfin as dlfn
from dolfin import inner, grad, div
dlfn.parameters["form_compiler"]["cpp_optimize"] = True
comm = dlfn.mpi_comm_world()
mpi_comm = dlfn.mpi_comm_world()
mpi_rank = dlfn.MPI.rank(mpi_comm)
print mpi_rank
#==============================================================================
# 1 for time-periodic case (Re=100) / 2 for fixed time interval (Re=100)
benchmark = 2
#==============================================================================
v_ref = 1.
l_ref = 0.1
rho_ref = 1.
mu_ref = 0.001
nu = mu_ref/rho_ref
t_ref = l_ref/v_ref
p_ref = rho_ref*v_ref**2
A_ref = dlfn.pi * (l_ref/2.)**2
reynolds = v_ref*l_ref/nu
#==============================================================================
if mpi_rank == 0:
    print "nondimensionalization:"
    print ""
    print "reference velocity    v_ref: ", v_ref, ", dim = [m/s]"
    print "reference length    l_ref: ", l_ref, ", dim = [m]"
    print "reference density    rho_ref: ", rho_ref, ", dim = [kg/m^3]"
    print "reference viscosity    mu_ref: ", mu_ref, ", dim = [kg/(m*s)]"
    print "kinematic viscosity    nu: ", nu, ", dim = [m^2/s]"
    print "refernce time    t_ref: ", t_ref, ", dim = [s]"
    print "reference pressure    p_ref: ", p_ref, ", dim = [kg/(m*s^2)]"
    print "---"
    print "reynolds number: ", reynolds, ", dim = [1]"
    print "---"
#==============================================================================
cylinder_id = 100
left_id = 101
wall_id = 102
right_id = 103
#==============================================================================
# mesh definition & refinement
mesh  = dlfn.Mesh("karman.xml")
facet_marker = dlfn.MeshFunctionSizet(mesh, "karman_facet_region.xml")
space_dim = mesh.topology().dim()
ncell = mesh.num_cells()
def integrateFluidStress(v, p, nu, space_dim):
    if space_dim == 2:
        cd_integral = -20.*(nu*v_ref*(1./l_ref**2)
                            * inner(dlfn.nabla_grad(v), dlfn.nabla_grad(vd))
                    + v_ref**2 * (1./l_ref)*inner(grad(v)*v, vd) 
                    - p_ref *(1./l_ref)*p*dlfn.div(vd))*dV
        cd = l_ref**2 * dlfn.assemble(cd_integral)
        cl_integral = -20.*(nu*v_ref*(1./l_ref**2) 
                            *inner(dlfn.nabla_grad(v), dlfn.nabla_grad(vl))
                    + v_ref**2 * (1./l_ref)* inner(grad(v)*v, vl) 
                    - p_ref *(1./l_ref)* p*dlfn.div(vl))*dV
        cl = l_ref**2 * dlfn.assemble(cl_integral)
    else:
        raise ValueError("space dimension unequal 2")
    return cd, cl
#==============================================================================
# list inizializing for postprocessing
t_list = []
iter_list = []
drag_coeff_list = []
lift_coeff_list = []
#==============================================================================
# cfl-number
cfl = 1.
u_max = 1.5
x_min = mesh.hmin()
#delta_t = cfl*x_min/u_max 
# specified time step (w/0 cfl-number)
delta_t = 1./8000.
d_t = dlfn.Constant(delta_t)
#==============================================================================
# element and function space definition
cell = mesh.ufl_cell()
# taylor-hood element
P2 = dlfn.VectorElement("CG", cell, 2)
P1 = dlfn.FiniteElement("CG", cell, 1)
mixed_space = P2 * P1
Wh = dlfn.FunctionSpace(mesh, mixed_space)
ndofs = Wh.dim()
ndofs_velocity = Wh.sub(0).dim()
ndofs_pressure = Wh.sub(1).dim()
#==============================================================================
# function space for validation of karman vortex street
Vh = dlfn.FunctionSpace(mesh, Wh.sub(0).ufl_element())
# function for drag coefficient
vd = dlfn.Function(Vh) # Zero everywhere!
vdBc = dlfn.DirichletBC(Vh, dlfn.Constant((1.,0.)), facet_marker, cylinder_id)
vdBc.apply(vd.vector()) # Set values on cylinder mantle
# function for lift coefficient
vl = dlfn.Function(Vh) # Zero everywhere!
vlBc = dlfn.DirichletBC(Vh, dlfn.Constant((0.,1.)), facet_marker, cylinder_id)
vlBc.apply(vl.vector()) # Set values on cylinder mantle
#==============================================================================
# definition of volume / surface element
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh)
#==============================================================================
# boundary conditions
n = dlfn.FacetNormal(mesh)
no_slip = dlfn.Constant((0.0, 0.0))
if benchmark == 1:
    v_inlet = dlfn.Expression(
        ("4.0 * U0 * x[1] * (H - x[1])/ pow(H, 2)", "0.0"),
        degree=2, H=4.1, U0=1.5)
else:
    v_inlet = dlfn.Expression(
        ("4.0 * U0* fabs(sin(pi*t*t_ref/8))*x[1] * (H - x[1])/ \
         pow(H, 2)", "0.0"), degree=2, H=4.1, U0=1.5, t=0., t_ref = 0.1)
# inlet bc
v_bc1 = dlfn.DirichletBC(Wh.sub(0), v_inlet, facet_marker, left_id)
# wall no slip
v_bc2 = dlfn.DirichletBC(Wh.sub(0), no_slip, facet_marker, wall_id)
# circle no slip
v_bc3 = dlfn.DirichletBC(Wh.sub(0), no_slip, facet_marker, cylinder_id)
bcs = [v_bc1, v_bc2, v_bc3]
raise ValueError()
#==============================================================================
# trial and test function
(w, q) = dlfn.TestFunctions(Wh)
(du, dp) = dlfn.TrialFunctions(Wh)
#------------------------------------------------------------------------------
sol = dlfn.Function(Wh)
sol0 = dlfn.Function(Wh)
v0, p0 = dlfn.split(sol0)
v, p = dlfn.split(sol)
#==============================================================================
# define nonlinear variational problem (nonsteady navier-stokes)
# implicit euler
F = (inner(v, w) 
    - inner(v0, w)
    + d_t*dlfn.Constant(1.0/reynolds)*(inner(grad(v), grad(w)))
    + d_t*inner(grad(v)*v, w) 
    - d_t*p*div(w) 
    - d_t*q*div(v)
    )*dV
J = dlfn.derivative(F, sol)
#==============================================================================
pvd_file = dlfn.File("results_{1}/Re_{0:d}/solution_velocity_Re_{0:d}_{1}.pvd".\
                     format(int(reynolds), "FEniCS", benchmark))
pvd_file2 = dlfn.File("results_{1}/Re_{0:d}/solution_pressure_Re_{0:d}_{1}.pvd".\
                     format(int(reynolds), "FEniCS", benchmark))
#==============================================================================
# time loop
dlfn.tic()
problem = dlfn.NonlinearVariationalProblem(F, sol, bcs, J)
solver = dlfn.NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["maximum_iterations"] = 20
t = 0.
iter = 0
if benchmark == 1:
    # inlet constant
    t_end = 300.
    while t < t_end - dlfn.DOLFIN_EPS:
            solver.solve()
            v, p = sol.split()
            cd, cl = integrateFluidStress(v, p, nu, space_dim)
            if iter%3200 == 0:
                if mpi_rank == 0:
                    print "t = {0:6.4f}".format(t)
                pvd_file << (v, t)
                pvd_file2 << (p, t)  
            drag_coeff_list.append(cd)
            lift_coeff_list.append(cl)
            iter_list.append(iter)
            t_dim = t*t_ref
            t_list.append(t_dim)
            # update for next iterate
            sol0.assign(sol)
            iter += 1
            t += delta_t
else:
    # inlet periodic
    t_end = 80.
    while t < t_end - dlfn.DOLFIN_EPS:
            solver.solve()
            v, p = sol.split()
            cd, cl = integrateFluidStress(v, p, nu, space_dim)
            if iter%3200 == 0:
                if mpi_rank == 0:
                    print "t = {0:6.4f}".format(t)
                pvd_file << (v, t)
                pvd_file2 << (p, t) 
            drag_coeff_list.append(cd)
            lift_coeff_list.append(cl)
            iter_list.append(iter)
            t_dim = t*t_ref
            t_list.append(t_dim)
            # update for next iterate
            sol0.assign(sol)
            v_inlet.t += delta_t
            iter += 1
            t += delta_t
if mpi_rank == 0:
    print 'total simulation time in seconds: ', dlfn.toc()
#==============================================================================
import numpy as np
header = "0.iteration 1.t in s 2.drag_coeff 3.lift_coeff"
np.savetxt('coeff_comp_8s_re{0}_implicit_euler_final.gz'.format(int(reynolds)),\
           np.column_stack((iter_list, t_list, drag_coeff_list,\
                            lift_coeff_list)), fmt='%1.4e',\
                            header = header)