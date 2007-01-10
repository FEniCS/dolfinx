# Import matplotlib
from pylab import *

# Import solution
from solution import *

# Read data for laps
from scipy.io.array_import import *
lorenz = read_array("lorenz.data")
tn    = lorenz[:, 0]
n0    = lorenz[:, 2]
n1    = lorenz[:, 3]
alpha = lorenz[:, 4]

# Plot solution
figure(1)
plot(t, u[:, 0], t, u[:, 1], t, u[:, 2])

# Plot laps
figure(2)
subplot(211)
plot(tn, n0)
xlabel('t')
ylabel('n1')

subplot(212)
plot(tn, n1)
xlabel('t')
ylabel('n2')

# Plot fraction n1 / n2
figure(3)
plot(tn, alpha)
xlabel('t')
ylabel('n1 / n2')

show()
