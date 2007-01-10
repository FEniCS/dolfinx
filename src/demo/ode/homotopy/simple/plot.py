# Import matplotlib
from pylab import *

# Load solutions
from solution_0 import *
(t0, u0, k0) = (t, u, k)
from solution_1 import *
(t1, u1, k1) = (t, u, k)
from solution_2 import *
(t2, u2, k2) = (t, u, k)

# Plot solutions
figure(1)
subplot(311)
plot(t0, u0[:, 0], t0, u0[:, 1])
xlabel('t')
ylabel('Re z, Im z')

subplot(312)
plot(t1, u1[:, 0], t1, u1[:, 1])
xlabel('t')
ylabel('Re z, Im z')

subplot(313)
plot(t2, u2[:, 0], t2, u2[:, 1])
xlabel('t')
ylabel('Re z, Im z')

# Plot time steps
figure(2)
subplot(311)
plot(t0, k0[:, 0])
xlabel('t')
ylabel('k')

subplot(312)
plot(t1, k1[:, 0])
xlabel('t')
ylabel('k')

subplot(313)
plot(t2, k2[:, 0])
xlabel('t')
ylabel('k')

# Plot paths
figure(3)

subplot(311)
plot(u0[:, 0], u0[:, 1])
xlabel('Re z')
ylabel('Im z')

subplot(312)
plot(u1[:, 0], u1[:, 1])
xlabel('Re z')
ylabel('Im z')

subplot(313)
plot(u2[:, 0], u2[:, 1])
xlabel('Re z')
ylabel('Im z')

show()
