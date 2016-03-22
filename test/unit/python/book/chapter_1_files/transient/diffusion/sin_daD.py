"""Temperature variations in the ground."""

from __future__ import print_function
from dolfin import *
import sys, numpy, time

# Usage:   sin_daD.py degree D   nx ny nz
# Example: sin_daD.py   1   1.5  4  40

# Create mesh and define function space
degree = int(sys.argv[1])
D = float(sys.argv[2])
W = D/2.0
divisions = [int(arg) for arg in sys.argv[3:]]
print('degree=%d, D=%g, W=%g, %s cells' % \
      (degree, D, W, 'x'.join(sys.argv[3:])))

d = len(divisions)  # no of space dimensions
if d == 1:
    mesh = Interval(divisions[0], -D, 0)
elif d == 2:
    mesh = RectangleMesh(Point(-W/2, -D), Point(W/2, 0), divisions[0], divisions[1])
elif d == 3:
    mesh = Box(-W/2, -W/2, -D, W/2, W/2, 0,
               divisions[0], divisions[1], divisions[2])
V = FunctionSpace(mesh, 'Lagrange', degree)

# Define boundary conditions

T_R = 0
T_A = 1.0
omega = 2*pi
# soil:
T_R = 10
T_A = 10
omega = 7.27E-5

T_0 = Expression('T_R + T_A*sin(omega*t)',
                 T_R=T_R, T_A=T_A, omega=omega, t=0.0, degree=2)

def surface(x, on_boundary):
    return on_boundary and abs(x[d-1]) < 1E-14

bc = DirichletBC(V, T_0, surface)

period = 2*pi/omega
t_stop = 5*period
#t_stop = period/4
dt = period/14
theta = 1

kappa_0 = 2.3  # KN/s (K=Kelvin, temp. unit), for soil
kappa_0 = 12.3  # KN/s (K=Kelvin, temp. unit)
kappa_1 = 1E+4
kappa_1 = kappa_0

kappa_str = {}
kappa_str[1] = 'x[0] > -D/2 && x[0] < -D/2 + D/4 ? kappa_1 : kappa_0'
kappa_str[2] = 'x[0] > -W/4 && x[0] < W/4 '\
               '&& x[1] > -D/2 && x[1] < -D/2 + D/4 ? '\
               'kappa_1 : kappa_0'
kappa_str[3] = 'x[0] > -W/4 && x[0] < W/4 '\
               'x[1] > -W/4 && x[1] < W/4 '\
               '&& x[2] > -D/2 && x[2] < -D/2 + D/4 ?'\
               'kappa_1 : kappa_0'

# Alternative way of defining the kappa function
class Kappa(Expression):
    def eval(self, value, x, **kwargs):
        """x: spatial point, value[0]: function value."""
        d = len(x)  # no of space dimensions
        material = 0  # 0: outside, 1: inside
        if d == 1:
            if -D/2. < x[d-1] < -D/2. + D/4.:
                material = 1
        elif d == 2:
            if -D/2. < x[d-1] < -D/2. + D/4. and \
               -W/4. < x[0] < W/4.:
                material = 1
        elif d == 3:
            if -D/2. < x[d-1] < -D/2. + D/4. and \
               -W/4. < x[0] < W/4. and -W/4. < x[1] < W/4.:
                material = 1
        value[0] = kappa_0 if material == 0 else kappa_1

# Physical parameters
kappa = Expression(kappa_str[d],
                   D=D, W=W, kappa_0=kappa_0, kappa_1=kappa_1, degree=0)

# soil:
rho = 1500
c = 1600

print('Thermal diffusivity:', kappa_0/rho/c)

# Define initial condition
T_1 = interpolate(Constant(T_R), V)

# Define variational problem
T = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
a = rho*c*T*v*dx + theta*dt*kappa*\
    inner(nabla_grad(v), nabla_grad(T))*dx
L = (rho*c*T_1*v + dt*f*v -
     (1-theta)*dt*kappa*inner(nabla_grad(v), nabla_grad(T)))*dx

A = assemble(a)
b = None  # variable used for memory savings in assemble calls

# Compute solution
T = Function(V)   # unknown at the new time level

# hack for plotting (first surface must span all values (?))
dummy = Expression('T_R - T_A/2.0 + 2*T_A*(x[%g]+D)' % (d-1),
                   T_R=T_R, T_A=T_A, D=D, dgeree=1)

# Make all plot commands inctive
import scitools.misc
plot = scitools.misc.DoNothing(silent=True)
# Need initial dummy plot
viz = plot(dummy, axes=True,
           title='Temperature', wireframe=False)
viz.elevate(-65)
#time.sleep(1)
viz.update(T_1)

import scitools.BoxField
start_pt = [0]*d; start_pt[-1] = -D  # start pt for line plot
import scitools.easyviz as ev

def line_plot():
    """Make a line plot of T along the vertical direction."""
    if T.ufl_element().degree() != 1:
        T2 = interpolate(T, FunctionSpace(mesh, 'Lagrange', 1))
    else:
        T2 = T
    T_box = scitools.BoxField.dolfin_function2BoxField(
            T2, mesh, divisions, uniform_mesh=True)
    #T_box = scitools.BoxField.update_from_dolfin_array(
    #        T.vector().array(), T_box)
    coor, Tval, fixed, snapped = \
            T_box.gridline(start_pt, direction=d-1)

    # Use just one ev.plot command, not hold('on') and two ev.plot
    # etc for smooth movie on the screen
    if kappa_0 == kappa_1:  # analytical solution?
        ev.plot(coor, Tval, 'r-',
                coor, T_exact(coor), 'b-',
                axis=[-D, 0, T_R-T_A, T_R+T_A],
                xlabel='depth', ylabel='temperature',
                legend=['numerical', 'exact, const kappa=%g' % kappa_0],
                legend_loc='upper left',
                title='t=%.4f' % t)
    else:
        ev.plot(coor, Tval, 'r-',
                axis=[-D, 0, T_R-T_A, T_R+T_A],
                xlabel='depth', ylabel='temperature',
                title='t=%.4f' % t)

    ev.savefig('tmp_%04d.png' % counter)
    time.sleep(0.1)

def T_exact(x):
    a = sqrt(omega*rho*c/(2*kappa_0))
    return T_R + T_A*exp(a*x)*numpy.sin(omega*t + a*x)

n = FacetNormal(mesh)  # for flux computation at the top boundary
flux = -kappa*dot(nabla_grad(T), n)*ds

t = dt
counter = 0
while t <= t_stop:
    print('time =', t)
    b = assemble(L, tensor=b)
    T_0.t = t
    help = interpolate(T_0, V)
    print('T_0:', help.vector().array()[0])
    bc.apply(A, b)
    solve(A, T.vector(), b)
    viz.update(T)
    #viz.write_ps('diff2_%04d' % counter)
    line_plot()

    total_flux = assemble(flux)
    print('Total flux:', total_flux)

    t += dt
    T_1.assign(T)
    counter += 1

viz = plot(T, title='Final solution')
viz.elevate(-65)
viz.azimuth(-40)
viz.update(T)

interactive()
