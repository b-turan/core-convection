#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import dolfin as dlfn
import mshr
from dolfin import inner, grad, div, dot
import numpy as np
#==============================================================================
dlfn.parameters["form_compiler"]["cpp_optimize"] = True
comm = dlfn.mpi_comm_world()
mpi_comm = dlfn.mpi_comm_world()
mpi_rank = dlfn.MPI.rank(mpi_comm)
print mpi_rank
print "The current version of FEniCS is ", dlfn.dolfin_version()
#==============================================================================
#======================   function definition   ===============================
#==============================================================================
def nusselt_fun(T, wall):
    # TO DO: get number of facet_marker automatically
    S = dlfn.assemble(1. * dA(wall))
    nusselt = dlfn.assemble(dot(grad(abs(T)), n)*dA(wall)) / S
    return nusselt
# append values to lists for post processor
def append_fun(v, p, T):
    # time
    t_list.append(t)
    # save values at time history point 1
    vx_list1.append(v(np.array(point1))[0])
    vy_list1.append(v(np.array(point1))[1])
    T_list1.append(T(np.array(point1)))
    # nusselt lists
    nu_inner_list.append(nu_inner)
    nu_outer_list.append(nu_outer)
    # root mean square of velocity
    v_metric_list.append(dlfn.sqrt(kin_e))
    # list of iteration
    iteration_list.append(cnt)
    return 
#==============================================================================
#==============================================================================
#==============================================================================
# time configuration
t = 0.
t_end = 150.
delta_t = 1e0
dt = dlfn.Constant(delta_t)
cnt = 0 # iteration counter
cfl = 1.# cfl-number
#==============================================================================
# dimensionless constants
rayleigh = 1e3
prandtl = 0.1
#==============================================================================
# ufl constants
Ra = dlfn.Constant(rayleigh)
Pr = dlfn.Constant(prandtl)
point1 = dlfn.Point()
#==============================================================================
# mesh creation
r_outer = 1. # outer radius
r_inner = r_outer/2. # inner radius
center = dlfn.Point(0., 0.)
domain = mshr.Circle(center, r_outer) \
       - mshr.Circle(center, r_inner)
mesh = mshr.generate_mesh(domain, 50)
space_dim = mesh.topology().dim()
n_cells = mesh.num_cells()
#==============================================================================
# subdomains for boundaries
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim-1)
facet_marker.set_all(0)
hmin = mesh.hmin() # length of smalles element
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
n = dlfn.FacetNormal(mesh)
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
    return dot(dot(grad(chi), phi), psi) * dV
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
#e_y = dlfn.Constant([0., 1.])
#==============================================================================
pvd_velocity = \
    dlfn.File("pvd_{0:1.1e}_{1:1.1e}_{2}s/solution_velocity.pvd".\
                         format(int(rayleigh), float(delta_t), int(t_end)))
#pvd_pressure = \
#    dlfn.File("pvd_{0:1.1e}_{1:1.1e}_{2}s/solution_pressure.pvd".\
#                         format(int(rayleigh), float(delta_t), int(t_end)))
pvd_temperature = \
    dlfn.File("pvd_{0:1.1e}_{1:1.1e}_{2}s/solution_temperature.pvd".\
                         format(int(rayleigh), float(delta_t), int(t_end)))
#==============================================================================
# time history points (post processing) 
point1 = [0.0, 0.75]  # [x-coordinate, y-coordinate]
point2 = [0.75, 0.0]
point3 = [0.0, -0.75]
point4 = [-0.75, 0.0]
#==============================================================================
# initialize lists (post processing)
t_list, iteration_list = [], []
T_list1 = []
vx_list1 = []
vy_list1 = []
nu_inner_list, nu_outer_list= [], []
v_metric_list = []
#==============================================================================
# define linear variational problem
# momentum equation
lhs_momentum = three_half * dot(dv, del_v) *  dV \
             + dt * dlfn.sqrt(Pr/Ra) * a_operator(dv, del_v) \
             - dt * b_operator(del_v, dp) \
             - dt * b_operator(dv, del_p)
rhs_momentum = two * dot(v0, del_v) * dV \
             - half * dot(v00, del_v) * dV \
             - dt * two * c_operator(v0, v0, del_v) \
             + dt * c_operator(v00, v00, del_v) \
# energy equation
lhs_energy = three_half * dot(dT, del_T) * dV \
           + dt / dlfn.sqrt(Ra*Pr) * a_operator(dT, del_T)
rhs_energy = two * dot(T0, del_T) * dV \
           - half * dot(T00, del_T) * dV
lhs = lhs_momentum + lhs_energy
# full problem
lhs = lhs_momentum + lhs_energy
rhs = rhs_momentum + rhs_energy
problem = \
        dlfn.LinearVariationalProblem(lhs, rhs, sol, bcs=bcs)
solver = dlfn.LinearVariationalSolver(problem)
#==============================================================================
# time loop
dlfn.tic()
# n_steps = 10000
while t < t_end:
    # solve linear problem
    solver.solve()
    v, p, T = sol.split(deepcopy=True)
    # write output
    if cnt%2 == 0:
        pvd_velocity << (v, t)
        #pvd_pressure << (p, t)
        pvd_temperature << (T, t)
        if mpi_rank == 0:
            print "t = {0:6.4f}".format(t)
            #print "Iteration: ", cnt
    if cnt%10 == 0:
        kin_e = dlfn.assemble(dlfn.Constant(1./(2.*A)) * dot(v, v) * dV)
        if kin_e > 1e12:
            raise ValueError("Instability occured!")
    nu_inner = nusselt_fun(T=T, wall=1) # inner core boundary nusselt number
    nu_outer = nusselt_fun(T=T, wall=2) # outer core boundary nusselt number
    append_fun(v, p, T)
    # update for next iteration
    sol00.assign(sol0)
    sol0.assign(sol)
    t += delta_t
    cnt += 1
if mpi_rank == 0:
    print 'elapsed simulation time in seconds: ', dlfn.toc()
#==============================================================================
header = "0. Iteration  " + "1. Time  " + "2. Temperature1  " + "3. v_x1  " \
         + "4. v_y1  "+  "5. nusselt inner  " + "6. nusselt outer  " \
         +  "delta_t = {0:1.1e}".format(float(delta_t))
np.savetxt('saved_values/ray_{0:1.1e}_dt_{1:1.1e}_{2}s.gz'
           .format(int(rayleigh), float(delta_t), int(t)), \
           np.column_stack((iteration_list, t_list, T_list1, 
                            vx_list1, vy_list1, nu_inner_list, nu_outer_list)),
                           fmt= '%1.6e', header = header)
           
           
           
# v_metric_list.append(dlfn.sqrt(kin_e))
# kin_e = dlfn.assemble(dlfn.Constant(1./(2.*A)) * dot(v, v) * dV)

