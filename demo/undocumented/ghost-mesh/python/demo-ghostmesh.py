#!/usr/bin/python
#
# very rough demo to test out ghost cells
# run with mpirun
#
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib as mpl
import numpy as np
import sys

parameters["ghost_mode"] = "shared_vertex"
# parameters["ghost_mode"] = "None"
parameters["reorder_cells_gps"] = True
parameters["reorder_vertices_gps"] = True

n = 0

if(len(sys.argv) == 2):
    try:
        n = int(sys.argv[1])
    except:
        n = 0

if(MPI.size(mpi_comm_world()) == 1):
    print "Only works with MPI"
    quit()

mpi_rank = MPI.rank(mpi_comm_world())

#parameters["mesh_partitioner"] = "ParMETIS"

mesh = UnitSquareMesh(32, 32)
# mesh = refine(M)


shared_vertices = mesh.topology().shared_entities(0).keys()
shared_cells = mesh.topology().shared_entities(mesh.topology().dim())

num_regular_vertices = mesh.topology().size(0) - mesh.topology().size_ghost(0)

ghost_vertices = range(num_regular_vertices, mesh.topology().size(0))

verts_note = []
if (n == 0):
    for k,val in mesh.topology().shared_entities(0).iteritems():
        vtx = Vertex(mesh, k)
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )
elif (n == 1):
    for i in range(mesh.num_vertices()):
        vtx = Vertex(mesh, i)
        val = vtx.global_index()
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )
else:
    for i in range(mesh.num_vertices()):
        vtx = Vertex(mesh, i)
        val = vtx.index()
        verts_note.append( (vtx.point().x(), vtx.point().y(), " "+str(val)) )

x,y = mesh.coordinates().transpose()

process_number = MPI.rank(mesh.mpi_comm())

cell_ownership = np.ones(mesh.num_cells(),dtype='int')*process_number
cell_owner = mesh.topology().cell_owner()
if len(cell_owner) > 0 :
    cell_ownership[-len(cell_owner):] = cell_owner

cells_store=[]
cells_note=[]
colors=[]
cmap=['red', 'green', 'yellow', 'purple', 'pink', 'grey', 'blue', 'brown']

idx = 0
for c in cells(mesh):
    xc=[]
    yc=[]
    for v in vertices(c):
        xc.append(v.point().x())
        yc.append(v.point().y())
    xavg = c.midpoint().x()
    yavg = c.midpoint().y()
    cell_str=str(c.index())
#    if c.index() in shared_cells.keys():
#        cell_str = str(shared_cells[c.index()])
#    else:
#        cell_str = str(c.index())
    cells_note.append((xavg, yavg, cell_str))
    cells_store.append(zip(xc,yc))

    colors.append(cmap[cell_ownership[c.index()]])
    idx += 1

num_regular_facets = mesh.topology().size(1) - mesh.topology().size_ghost(1)
facet_note = []
shared_facets = mesh.topology().shared_entities(1)
for f in facets(mesh):
    if (f.num_global_entities(2) == 2):
        color='#ffff88'
    else:
        color='#ff88ff'
    if (not f.is_ghost()):
        if (f.num_global_entities(2) == 2):
            color='#ffff00'
        else:
            color='#ff00ff'

    if (n < 3):
        facet_note.append((f.midpoint().x(), f.midpoint().y(), f.global_index(), color))
    elif (n == 3):
        facet_note.append((f.midpoint().x(), f.midpoint().y(), f.index(), color))
    else:
        if (f.index() in shared_facets.keys()):
            facet_note.append((f.midpoint().x(), f.midpoint().y(), shared_facets[f.index()], color))

fig, ax = plt.subplots()

# Make the collection and add it to the plot.
coll = PolyCollection(cells_store, facecolors=colors, edgecolors='#cccccc')
ax.add_collection(coll)

plt.plot(x, y, marker='o', color='black', linestyle='none')
plt.plot(x[shared_vertices], y[shared_vertices], marker='o', color='green', linestyle='none')
plt.plot(x[ghost_vertices], y[ghost_vertices], marker='o', color='yellow', linestyle='none')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

plt.xlim((xlim[0] - 0.1, xlim[1] + 0.1))
plt.ylim((ylim[0] - 0.1, ylim[1] + 0.1))

for note in cells_note:
    plt.text(note[0], note[1], note[2], verticalalignment='center',
             horizontalalignment='center', size=8)

for note in verts_note:
    plt.text(note[0], note[1], note[2], size=8, verticalalignment='center')

#for note in facet_note:
#    plt.text(note[0], note[1], note[2], size=8, verticalalignment='center', backgroundcolor=note[3])

#plt.show()


"""
# Create Poisson problem on mesh
V = FunctionSpace(mesh, "Lagrange", 2)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

A = assemble(a)
print "Matrix norm:", A.norm("frobenius")
b = assemble(L)
print "Vector norm:", b.norm("l2")

# Compute solution
u = Function(V)
solve(a == L, u, bc)
print "Solution norm: ", u.vector().norm("l2")
"""

V = FunctionSpace(mesh, "CG", 2)

# Define Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Source(Expression):
    def eval(self, values, x):
        values[0] = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define normal component, mesh size and right-hand side
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
n = FacetNormal(mesh)
f = Source()

# Penalty parameter
alpha = Constant(8.0)

# Define bilinear form
a = inner(div(grad(u)), div(grad(v)))*dx \
  - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
  - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
  + alpha/h_avg*inner(jump(grad(u),n), jump(grad(v),n))*dS

# Define linear form
L = f*v*dx

A = assemble(a)
bc.apply(A)
print "Matrix norm:", A.norm("frobenius")
b = assemble(L)
print "Vector norm (0):", b.norm("l2")
bc.apply(b)
print "Vector norm (1):", b.norm("l2")


# Solve variational problem
u = Function(V)
solve(a == L, u, bc)
print "Solution vector: ", u.vector().norm("l2")
