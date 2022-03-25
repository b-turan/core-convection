#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import dolfin as dlfn
from dolfin import inner, grad, div, dot
dlfn.parameters["form_compiler"]["cpp_optimize"] = True
#==============================================================================
v_ref = 1.
l_ref = 0.1
rho_ref = 1.
mu_ref = 0.001
nu = mu_ref/rho_ref
t_ref = l_ref/v_ref
p_ref = rho_ref*v_ref**2
A_ref = dlfn.pi * (l_ref/2.)**2
#==============================================================================
# dimensionless constants
reynolds = v_ref*l_ref/nu
#==============================================================================
# ufl constants
Re = dlfn.Constant(reynolds)
#==============================================================================
print "-" * 15 + " IMEX scheme for Navier-Stokes flow " + "-" * 15
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
#==============================================================================
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
# minimal mesh length
h_min = mesh.hmin()
# time step
delta_t = 1./1.
dt = dlfn.Constant(delta_t)
#==============================================================================
# list inizializing for postprocessing
t_list = []
iter_list = []
drag_coeff_list = []
lift_coeff_list = []
#==============================================================================
# element and function space definition
cell = mesh.ufl_cell()
# taylor-hood element
elemV = dlfn.VectorElement("CG", cell, 2)
elemP = dlfn.FiniteElement("CG", cell, 1)
mixedElem = dlfn.MixedElement([elemV, elemP])
Wh = dlfn.FunctionSpace(mesh, mixedElem)
ndofs = Wh.dim()
ndofs_velocity = Wh.sub(0).dim()
ndofs_pressure = Wh.sub(1).dim()
#==============================================================================
# function space for validation of karman vortex street
Vh = dlfn.FunctionSpace(mesh, Wh.sub(0).ufl_element())
# function for drag coefficient
vd = dlfn.Function(Vh) #Zero everywhere!
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
null_vector = dlfn.Constant((0.0, 0.0))
# inlet velocity
v_inlet = dlfn.Expression(
    ("4.0 * U0* fabs(sin(pi*t*t_ref/8))*x[1] * (H - x[1])/ \
     pow(H, 2)", "0.0"), degree=2, H=4.1, U0=1.5, t=0., t_ref = 0.1)
# inlet bc
v_bc1 = dlfn.DirichletBC(Wh.sub(0), v_inlet, facet_marker, left_id)
# wall no slip
v_bc2 = dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, wall_id)
# circle no slip
v_bc3 = dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, cylinder_id)
# collection of bcs
bcs = [v_bc1, v_bc2, v_bc3]
#==============================================================================
# trial and test function
(del_v, del_p) = dlfn.TestFunctions(Wh)
(dv, dp) = dlfn.TrialFunctions(Wh)
#==============================================================================
# solution functions
sol = dlfn.Function(Wh)
sol0 = dlfn.Function(Wh)
sol00 = dlfn.Function(Wh)
v0, p0 = dlfn.split(sol0)
v00, p00 = dlfn.split(sol00)
#==============================================================================
# define auxiliary operators
def a_operator(phi, psi):
    return inner(grad(phi), grad(psi)) / Re  * dV
def b_operator(phi, psi):
    return div(phi) * psi * dV
def c_operator(phi, chi, psi):
    return  dot(grad(chi)*phi, psi) * dV
#==============================================================================
# ATTENTION: beta is called c in Ascher(1995)
beta = 0.
gamma = 1.
# coefficents
a = [dlfn.Constant(gamma + 0.5), 
     dlfn.Constant(gamma - 0.5),
     dlfn.Constant(2. * gamma)]
b = [None, 
     dlfn.Constant(gamma),
     dlfn.Constant(gamma + 1.0)]
c = [dlfn.Constant(gamma + beta / 2.),
     dlfn.Constant(beta / 2.0),
     dlfn.Constant(1. - gamma - beta)]
#==============================================================================
two = dlfn.Constant(2.)
half = dlfn.Constant(1./2.)
three_half = dlfn.Constant(3./2.)
#==============================================================================
# define linear variational problem
lhs = three_half / dt * dot(dv, del_v) *  dV \
    + a_operator(dv, del_v) \
    + two * c_operator(v0, dv, del_v) \
    - c_operator(v00, dv, del_v) \
    - b_operator(del_v, dp) \
    - b_operator(dv, del_p)
rhs = \
      two / dt * dot(v0, del_v) *  dV \
    - half / dt * dot(v00, del_v) *  dV
#==============================================================================
pvd_velocity = dlfn.File("results_{1}/Re_{0:d}/solution_velocity_Re_{0:d}.pvd".\
                     format(int(reynolds), "FEniCS"))
pvd_pressure = dlfn.File("results_{1}/Re_{0:d}/solution_pressure_Re_{0:d}.pvd".\
                     format(int(reynolds), "FEniCS"))
#==============================================================================
# time loop
dlfn.tic()
t = 0.
cnt = 0
n_steps = 5000
t_end = 80.
while t < t_end: #and cnt <= n_steps:
    problem = dlfn.LinearVariationalProblem(lhs, rhs, sol, bcs=bcs)
    solver = dlfn.LinearVariationalSolver(problem)
    solver.solve()
    v, p = sol.split()
    if cnt % 3200 == 0:
        print "t = {0:6.4f}".format(t)
        pvd_velocity << (v, t)
        pvd_pressure << (p, t)
    cd, cl = integrateFluidStress(v, p, nu, space_dim)
    drag_coeff_list.append(cd)
    lift_coeff_list.append(cl)
    iter_list.append(cnt)
    t_dim = t*t_ref
    t_list.append(t_dim)
    # update for next iteration
    sol00.assign(sol0)
    sol0.assign(sol)
    t += delta_t
    v_inlet.t += delta_t
    cnt += 1
print 'total simulation time in seconds: ', dlfn.toc()
#==============================================================================
import numpy as np
header = "0.iteration 1.t in s 2.drag_coeff 3.lift_coeff"
np.savetxt('coeff_comp_8s_re{0}_imex_final.gz'.format(int(reynolds)),\
           np.column_stack((iter_list, t_list, drag_coeff_list,\
                            lift_coeff_list)), fmt='%1.4e',\
                            header = header)