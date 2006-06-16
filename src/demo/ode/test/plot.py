# Import solution
from solution import *

# Import matplotlib
from pylab import *

# Plot the two components of the solution
plot(t, u[:,0], 'b')
hold(True)
plot(t, u[:,1], 'r')
grid(True)
xlabel('t')
ylabel('U(t)')
title('Harmonic oscillator')
show()
