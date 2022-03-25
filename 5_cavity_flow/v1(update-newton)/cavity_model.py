#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
#######################################################################
##           Solving steady state Navier-Stokes-Equation             ##
##                       with Hybrid-Solver                          ##
##                (Picard-Update + Newtons method)                   ## 
#######################################################################
"""
import dolfin as dlfn
from dolfin import inner, grad, div, derivative
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 18})
#==============================================================================
# dimensionless constant
reynolds = 5000. # reynolds-zahl
Re = dlfn.Constant(reynolds)    
validierung = True  # validierungsplot erstellen?
#==============================================================================
print "---"
print "Reynolds number: ", reynolds
print "---"
print "Validierung?: ", validierung
#==============================================================================
# mesh definition & refinement 
mesh = dlfn.Mesh("square_mod.xml")
for x in range(5): # coarse grid
    mesh = dlfn.refine(mesh)
space_dim = mesh.topology().dim()
ncell = mesh.num_cells()
h_min = mesh.hmin()
#==============================================================================
# element and functionSpace definition
cell = mesh.ufl_cell()
# taylor-Hood element
P1 = dlfn.VectorElement("CG", cell, 2)
P2 = dlfn.FiniteElement("CG", cell, 1) 
mixed_space = P1 * P2
W = dlfn.FunctionSpace(mesh, mixed_space)
ndofs = W.dim()
ndofs_velocity = W.sub(0).dim()
ndofs_pressure = W.sub(1).dim()
#==============================================================================
# subdomains for boundaries
gamma_left = dlfn.CompiledSubDomain("near(x[0], 0.) && on_boundary")
gamma_right = dlfn.CompiledSubDomain("near(x[0], 1.) && on_boundary")
gamma_top = dlfn.CompiledSubDomain("near(x[1], 1.) && on_boundary")
gamma_bottom = dlfn.CompiledSubDomain("near(x[1], 0.) && on_boundary")
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim -1)
facet_marker.set_all(0)
gamma_left.mark(facet_marker, 1)
gamma_right.mark(facet_marker, 1)
gamma_bottom.mark(facet_marker, 1)
gamma_top.mark(facet_marker, 2)
#dlfn.File('mesh_lvl1.pvd') << facet_marker
#raise ValueError()
#==============================================================================
# definition of volume / surface element
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh, subdomain_data = facet_marker)
#==============================================================================
# trial and testfunction
(v, q) = dlfn.TestFunctions(W)
(du, dp) = dlfn.TrialFunctions(W)
w = dlfn.Function(W)
u, p = dlfn.split(w)
#==============================================================================
# define linearized weak form / variational problem (steady state Navier-Stokes)
f = dlfn.Constant((0.0, 0.0))
F = ((1./Re) * inner(grad(u), grad(v)) + inner(grad(u)*u, v) \
        - p*div(v) - q*div(u))*dV - inner(f, v)*dV
#==============================================================================
# Picard's Jacobian
Jpicard = ((1./Re)*inner(grad(du), grad(v)) + inner(grad(du)*u, v) \
         - dp*div(v) - q*div(du))*dV
#==============================================================================
#boundary conditions
no_slip = [0, 0]
slip = [1.0, 0]
no_slip = dlfn.DirichletBC(W.sub(0), no_slip, facet_marker, 1)
slip = dlfn.DirichletBC(W.sub(0), slip, facet_marker, 2)
bcs = [no_slip, slip]
#==============================================================================
# hybrid's first solver: picard's iteration
problem = dlfn.NonlinearVariationalProblem(F, w, bcs, J=Jpicard)
picard_solver = dlfn.NonlinearVariationalSolver(problem)
picard_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-5
picard_solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-5
picard_solver.parameters["newton_solver"]["maximum_iterations"] = 100
picard_solver.solve()
#==============================================================================
# hybrid's second solver: newton's method
J = derivative(F, w)
problem = dlfn.NonlinearVariationalProblem(F, w, bcs, J=J)
newton_solver = dlfn.NonlinearVariationalSolver(problem)
newton_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-11
newton_solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
newton_solver.parameters["newton_solver"]["maximum_iterations"] = 40
newton_solver.solve()
(usol, psol) = w.split()
#==============================================================================
#==============================================================================
#==============================================================================
# second run (finer grid)
mesh = dlfn.Mesh("square_mod.xml")
for x in range(6):
    mesh = dlfn.refine(mesh)
space_dim = mesh.topology().dim()
ncell = mesh.num_cells()
h_min = mesh.hmin()
#==============================================================================
# element and functionSpace definition
cell = mesh.ufl_cell()
# taylor-Hood element
P1 = dlfn.VectorElement("CG", cell, 2)
P2 = dlfn.FiniteElement("CG", cell, 1) 
mixed_space = P1 * P2
W = dlfn.FunctionSpace(mesh, mixed_space)
ndofs = W.dim()
ndofs_velocity = W.sub(0).dim()
ndofs_pressure = W.sub(1).dim()
(v, q) = dlfn.TestFunctions(W)
(du, dp) = dlfn.TrialFunctions(W)
w = dlfn.Function(W)
u, p = dlfn.split(w)
#==============================================================================
# subdomains for boundaries
gamma_left = dlfn.CompiledSubDomain("near(x[0], 0.) && on_boundary")
gamma_right = dlfn.CompiledSubDomain("near(x[0], 1.) && on_boundary")
gamma_top = dlfn.CompiledSubDomain("near(x[1], 1.) && on_boundary")
gamma_bottom = dlfn.CompiledSubDomain("near(x[1], 0.) && on_boundary")
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim -1)
facet_marker.set_all(0)
gamma_left.mark(facet_marker, 1)
gamma_right.mark(facet_marker, 1)
gamma_bottom.mark(facet_marker, 1)
gamma_top.mark(facet_marker, 2)
dlfn.File('subdomains.pvd') << facet_marker
#raise ValueError()
#==============================================================================
# definition of volume / surface element
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh, subdomain_data = facet_marker)
#==============================================================================
# trial and testfunction
(v, q) = dlfn.TestFunctions(W)
(du, dp) = dlfn.TrialFunctions(W)
w = dlfn.Function(W)
u, p = dlfn.split(w)
#==============================================================================
# define linearized weak form / variational problem (steady state Navier-Stokes)
f = dlfn.Constant((0.0, 0.0))
F = ((1./Re) * inner(grad(u), grad(v)) + inner(grad(u)*u, v) \
        - p*div(v) - q*div(u))*dV - inner(f, v)*dV
#==============================================================================
# Picard's Jacobian
Jpicard = ((1./Re)*inner(grad(du), grad(v)) + inner(grad(du)*u, v) \
         - dp*div(v) - q*div(du))*dV
#==============================================================================
#boundary conditions
no_slip = [0, 0]
slip = [1.0, 0]
no_slip = dlfn.DirichletBC(W.sub(0), no_slip, facet_marker, 1)
slip = dlfn.DirichletBC(W.sub(0), slip, facet_marker, 2)
bcs = [no_slip, slip]
#==============================================================================
# hybrid's first solver: picard's iteration
problem = dlfn.NonlinearVariationalProblem(F, w, bcs, J=Jpicard)
picard_solver = dlfn.NonlinearVariationalSolver(problem)
picard_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-5
picard_solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-5
picard_solver.parameters["newton_solver"]["maximum_iterations"] = 100
picard_solver.solve()
#==============================================================================
# hybrid's second solver: newton's method
J = derivative(F, w)
problem = dlfn.NonlinearVariationalProblem(F, w, bcs, J=J)
newton_solver = dlfn.NonlinearVariationalSolver(problem)
newton_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-11
newton_solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
newton_solver.parameters["newton_solver"]["maximum_iterations"] = 40
newton_solver.solve()
(usol2, psol2) = w.split()
#==============================================================================
#==============================================================================                              
                                  # Post-Processing
#==============================================================================
y = np.linspace(0,1,100)
x = 0.5
usol_x = [0.] * len(y)
usol_x2 = [0.] * len(y)
i=0
while i < len(y):
    usol_x[i] = usol(np.array([x, y[i]]))[0]
    usol_x2[i] = usol2(np.array([x, y[i]]))[0]
    i+=1
#==============================================================================
# reference values ghia et al.
y_ghia = [0., 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000,\
          0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
ghia_400 = [0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, \
            -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892,\
            0.61756, 0.68439, 0.75837, 1.0000]
ghia_1000 = [0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, \
             -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604,\
             0.51117, 0.57492, 0.65928, 1.00000]
ghia_5000 = [0.00000, -0.41165, -0.42901, -0.43643, -0.40435, -0.33050, \
             -0.22855, -0.07404, -0.03039, 0.08183, 0.20087, 0.33556, 0.46036,\
             0.45992, 0.46120, 0.48223, 1.00000]
#==============================================================================
# comparison
if validierung:
    list_reynolds = [400, 1000, 5000]
    if int(reynolds) == list_reynolds[0] or int(reynolds) == list_reynolds[1] \
    or int(reynolds) == list_reynolds[2]:
        list_validierung = [ghia_400, ghia_1000, ghia_5000]
        for i in range(3):
            if list_reynolds[i] == int(reynolds):
                ghia_ref = list_validierung[i]
                break
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(usol_x, y, "--", 
                label = r'\textit{Hybrid-Solver} ($64 \times 64$)')
        ax.plot(usol_x2, y, color = "black",
                label = r'\textit{Hybrid-Solver} ($128 \times 128$)')
        ax.plot(ghia_ref, y_ghia, "o",color='red', \
                label='Referenz [Ghia u. a. 1982]')
        ax.set_xlabel(r'$v_x(x = 0{,}5)$', fontsize = 22)
        ax.set_ylabel(r'$y$-Koordinate', fontsize = 22)
        ax.grid(True)
        ax.legend(loc='best', shadow=True, fontsize = 18)
        fig.savefig('Validierungsplots/ref_ghia_{0:d}_hybrid.pdf'.\
                    format(list_reynolds[i]), bbox_inches='tight')
    else:
        print ""
        print "Ein Validierungsplot konnte nicht vorgenommen werden, da "
        print "die gewählte Reynoldszahl", reynolds, "nicht mit denen aus den "
        print "Validierungsbeispielen von Ghia übereinstimmt"
        print "Um eine Validierung (mit Ghia) durchführen zu können, muss die"
        print "Reynoldszahl auf 400/1000/5000 gesetzt werden"
#==============================================================================
pvd_file = dlfn.File("results/Re_{0:d}/solution_velocity_Re_{0:d}_{1}.pvd".\
                     format(int(reynolds) , "FEniCS"))
pvd_file << usol
pvd_file = dlfn.File("results/Re_{0:d}/solution_pressure_Re_{0:d}_{1}.pvd".\
                     format(int(reynolds), "FEniCS"))
pvd_file << psol