"""
FEniCS tutorial demo program:
The Poisson equation with a variable coefficient.

-div(p*grad(u)) = f on the unit square.
u = u0 on x=0,
u0 = u = 1 + x^2 + 2y^2, p = x + y, f = -8x - 10y.
"""

from __future__ import print_function
from dolfin import *
from six.moves import input
import numpy
plot = lambda *args, **kwargs: None

# Create mesh and define function space
nx = 2
ny = 3
nx = 10
ny = 10
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

u0_boundary = DirichletBoundary()
bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
p = Expression('x[0] + x[1]', degree=1)
f = Expression('-8*x[0] - 10*x[1]', degree=1)
a = p*inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Compute gradient
V_g = VectorFunctionSpace(mesh, 'Lagrange', 1)
v = TestFunction(V_g)
w = TrialFunction(V_g)

a = inner(w, v)*dx
L = inner(-p*grad(u), v)*dx
flux = Function(V_g)
solve(a == L, flux)

# Alternative
flux = project(-p*grad(u), VectorFunctionSpace(mesh, 'Lagrange', 1))

plot(u, title='u')
plot(flux, title='flux field')

flux_x, flux_y = flux.split(deepcopy=True)  # extract components
plot(flux_x, title='x-component of flux (-p*grad(u))')
plot(flux_y, title='y-component of flux (-p*grad(u))')
plot(mesh)

# Alternative computation of the flux
flux2 = project(-p*grad(u), VectorFunctionSpace(mesh, 'Lagrange', 1))

print(mesh)

# Dump solution and flux to the screen with errors
u_array = u.vector().array()
flux_x_array = flux_x.vector().array()  # ok if deepcopy
flux_y_array = flux_y.vector().array()
# if not deepcopy of flux_x, flux_y:
#q = flux.vector().array()
#q.shape = (2, len(q)/2)
#flux_x_array = q[0,:]
#flux_y_array = q[1,:]
if mesh.num_cells() < 1600:
    coor = mesh.coordinates()
    for i in range(len(u_array)):
        x, y = coor[i]
        print('Node (%.3f,%.3f): u = %.4f (%9.2e), '\
              'flux_x = %.4f  (%9.2e), flux_y = %.4f  (%9.2e)' % \
              (x, y, u_array[i], 1 + x**2 + 2*y**2 - u_array[i],
               flux_x_array[i], -(x+y)*2*x - flux_x_array[i],
               flux_y_array[i], -(x+y)*4*y - flux_y_array[i]))

# Plot solution and flux
import scitools.BoxField
import scitools.easyviz as ev
X = 0; Y = 1; Z = 2
# Note: avoid * import from easyviz as DOLFIN and has already
# defined plot, figure, mesh

u2 = u if u.ufl_element().degree() == 1 else \
     interpolate(u, FunctionSpace(mesh, 'Lagrange', 1))
# alternatively: interpolate onto a finer mesh for higher degree elements
u_box = scitools.BoxField.dolfin_function2BoxField(
        u2, mesh, (nx,ny), uniform_mesh=True)

# Write out u at mesh point (i,j)
i = nx; j = ny
print('u(%g,%g)=%g' % (u_box.grid.coor[X][i],
                       u_box.grid.coor[Y][j],
                       u_box.values[i,j]))
ev.contour(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values,
           14, savefig='tmp0.eps', title='Contour plot of u',
           clabels='on')
ev.figure()
ev.surf(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values,
        shading='interp', colorbar='on',
        title='surf plot of u', savefig='tmp3.eps')
ev.figure()
ev.mesh(u_box.grid.coorv[X], u_box.grid.coorv[Y], u_box.values,
        title='mesh plot of u', savefig='tmp4.eps')

# Extract and plot u along the line y=0.5
start = (0,0.5)
x, uval, y_fixed, snapped = u_box.gridline(start, direction=X)
if snapped:
    print('Line at %s adjusted (snapped) to y=%g' % (start, y_fixed))
ev.figure()
ev.plot(x, uval, 'r-', title='Solution',
        legend='finite element solution')

# Plot the numerical (projected) and exact flux along this line
ev.figure()
flux2_x = flux_x if flux_x.ufl_element().degree() == 1 else \
          interpolate(flux_x, FunctionSpace(mesh, 'Lagrange', 1))
flux_x_box = scitools.BoxField.dolfin_function2BoxField(
        flux2_x, mesh, (nx,ny), uniform_mesh=True)
x, fluxval, y_fixed, snapped = \
        flux_x_box.gridline(start, direction=0)
y = y_fixed
flux_x_exact = -(x + y)*2*x
ev.plot(x, fluxval, 'r-',
        x, flux_x_exact, 'b-',
        legend=('numerical (projected) flux', 'exact flux'),
        title='Flux in x-direction (at y=%g)' % y_fixed,
        savefig='tmp1.eps')


# Plot flux along a line with many points also in the interior of
# the elements
# NOTE (FIXME): Strange artifacts at the end of the line!!!
n = 101
#n = 10
x = numpy.linspace(0, 1, n)
y = numpy.zeros(x.size) + 0.5  # y[i] = 0.5
xy_coor = numpy.array([x, y]).transpose()
#print 'xy_coor:', xy_coor
flux_x_line = numpy.zeros(x.size)
for i in range(len(flux_x_line)):
    flux_x_line[i] = flux_x(xy_coor[i])
flux_x_exact = -(x + y_fixed)*2*x
ev.figure()
ev.plot(x, flux_x_line, 'r-',
        x, flux_x_exact, 'b-',
        legend=('projected flux evaluated at %d points' % n, 'exact flux'),
        title='Flux at y=%g' % y[0],
        safefig='tmp2.eps')

# Verification
u_e = interpolate(u0, V)
u_e_array = u_e.vector().array()
print('Max error:', numpy.abs(u_e_array - u_array).max())

#interactive()
input('Press Return: ')  # some curve plot engines need this for a lasting plot on the screen
