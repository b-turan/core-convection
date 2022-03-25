#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:00:00 2018

@author: Berkant 

#######################################################################
##           Solving steady state Navier-Stokes-Equation             ##
##                       with Hybrid-Solver                          ##
##                 (Picards-Fixpunkt + Newtons method)               ## 
#######################################################################
"""
import dolfin as dlfn
from dolfin import inner, grad, div
import numpy as np
import matplotlib.pyplot as plt

reynolds = 100.                    # Reynoldszahl (float)
Re = dlfn.Constant(reynolds)    
validierung = False                 # Validierungsplot erstellen? 
#==============================================================================
print "---"
print "Reynolds number: ", reynolds
print "---"
print "Validierung?: ", validierung
print "---"
#==============================================================================
#Mesh definition & refinement
mesh = dlfn.Mesh("square_mod.xml")
for x in range(3):
    mesh = dlfn.refine(mesh)
space_dim = mesh.topology().dim()
ncell = mesh.num_cells()
dlfn.plot(mesh)
#==============================================================================
#Element and FunctionSpace definition
cell = mesh.ufl_cell()
#Taylor-Hood element
P1 = dlfn.VectorElement("CG", cell, 2)
P2 = dlfn.FiniteElement("CG", cell, 1) 
mixed_space = P1 * P2
W = dlfn.FunctionSpace(mesh, mixed_space)
ndofs = W.dim()
ndofs_velocity = W.sub(0).dim()
ndofs_pressure = W.sub(1).dim()
#==============================================================================
#Subdomains for boundaries
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
#==============================================================================
# definition of volume / surface element (not explicitly neccessary)
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh, subdomain_data = facet_marker)
#==============================================================================
#Trial- & Testfunction
(v, q) = dlfn.TestFunctions(W)
(u, p) = dlfn.TrialFunctions(W)
w = dlfn.Function(W)
#==============================================================================
# Initial guess to linearize variational problem
start_velocity = dlfn.Constant((0., 0.))
u_k = dlfn.interpolate(start_velocity, dlfn.FunctionSpace(mesh, P1))
#==============================================================================
# Define linearized weak form / variational problem (Picard Iteration)
f = dlfn.Constant((0.0, 0.0))
a = ((1/Re) * (inner(grad(u), grad(v))) + inner(grad(u)*u_k, v) \
     - p*div(v) - q*div(u))*dV
L = inner(f, v)*dV
#==============================================================================
# boundary conditions
no_slip = [0, 0]
slip = [1.0, 0]
no_slip = dlfn.DirichletBC(W.sub(0), no_slip, facet_marker, 1)
slip = dlfn.DirichletBC(W.sub(0), slip, facet_marker, 2)
bcs = [no_slip, slip]
#==============================================================================
# Hybrid's 1st solver: Picard's Iteration
iter = 0                # Iterationsbeginn
max_iter = 5            # maximal Iterationen
if reynolds > 2000:
    max_iter = 50
problem = dlfn.LinearVariationalProblem(a, L,  w, bcs)
problem_solver = dlfn.LinearVariationalSolver(problem)
problem_solver.parameters["linear_solver"] = "direct"
while iter < max_iter:
    iter += 1
    problem_solver.solve()    
    (usol, psol) = w.split(deepcopy=True)
    u_k.assign(usol)
    p = psol
#==============================================================================
# Hybrid's 2nd solver: Newton's Method
(u, p)  = dlfn.split(w)
F = ((1/Re) * (inner(grad(u), grad(v))) + inner(grad(u)*u, v) \
     - p*div(v) - q*div(u))*dV - inner(f, v)*dV
from dolfin import derivative
J = derivative(F, w)
problem = dlfn.NonlinearVariationalProblem(F, w, bcs, J=J)
solver = dlfn.NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["maximum_iterations"] = 50
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-12
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-13
niter, _ = solver.solve()
print "Number of Picard-Iteration: ", iter
print "Number of Newton-iterations: ", niter
(usol, psol) = w.split()
#==============================================================================
# Post-Processing
# Plot of horizontal velocity at vertical line through geometric center (x=0.5)
y = np.linspace(0,1,100)
x = 0.5
usol_x = [0.] * len(y)
i=0
while i < len(y):
    usol_x[i] = usol(np.array([x, y[i]]))[0]
    i+=1
#==============================================================================
# Reference Ghia et al. (horizontale Geschwindigkeitskomponente bei x=0.5)
y_ghia = [0., 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000,\
          0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
ghia_400 = [0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, \
            -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892,\
            0.61756, 0.68439, 0.75837, 1.0000]
ghia_1000 = [0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, \
             -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604,\
             0.51117, 0.57492, 0.65928, 1.00000]
ghia_5000 = [0.00000, -0.41165, -0.42901, -0.43643, -0.40435, -0.33050, \
             -0.22855, -0.07404, 0.03039, 0.081832, 0.20087, 0.33556, 0.46036,\
             0.45992, 0.46120, 0.48223, 1.00000]
#==============================================================================
# Validierungsplot
if validierung:
    list_reynolds = [400, 1000, 5000]
    if int(reynolds) == list_reynolds[0] or int(reynolds) == list_reynolds[1] \
    or int(reynolds) == list_reynolds[2]:
        list_validierung = [ghia_400, ghia_1000, ghia_5000]
        for i in range(3):
            if list_reynolds[i] == int(reynolds):
                ghia_ref = list_validierung[i]
                break
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(usol_x, y, label = 'Hybrid-Solver (Fixpunkt)')
        ax.plot(ghia_ref, y_ghia, "o",color='red', label='Reference [Ghia82]')
        ax.set_title('horizontale Geschwindigkeit bei x=0.5, Re = {0:d}'.\
                     format(int(reynolds)), fontsize=10)
        ax.set_xlabel('horizontale Geschwindigkeit')
        ax.set_ylabel('y-Koordinate')
        ax.grid(True)
        ax.legend(loc=5)
        fig.savefig('Validierungsplots/ref_ghia_{0:d}_fixpunkt.pdf'.\
                    format(list_reynolds[i]), bbox_inches='tight')
        plt.close(fig)    # close the figure
    else:
        print ""
        print "Ein Validierungplot konnte nicht vorgenommen werden, da "
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