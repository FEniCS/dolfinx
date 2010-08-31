# Import solution
from solution import *

# Import matplotlib
from pylab import *

# Plot solution
figure(1)
plot(t, u[:,0], t, u[:,1])
xlabel('t')
ylabel('U(t)')
title('Harmonic oscillator')

# Plot solution
figure(2)
plot(t, k)
xlabel('t')
ylabel('k(t)')
title('Time steps')

show()
