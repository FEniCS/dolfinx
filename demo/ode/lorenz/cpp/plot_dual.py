# Import matplotlib
from pylab import *

# Import solution
from solution_dual import *

# Plot solution
figure(1)
semilogy(t, u[:, 0], t, u[:, 1], t, u[:, 2])
xlabel('t')
ylabel('psi(t)')
title('Lorenz dual')

show()
