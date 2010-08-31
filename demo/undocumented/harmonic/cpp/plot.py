# Import solution
from solution import *

# Import matplotlib
from pylab import *

# Plot solution
plot(t, u[:,0], t, u[:,1])
xlabel('t')
ylabel('U(t)')
title('Harmonic oscillator')
show()
