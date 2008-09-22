# Import matplotlib
from pylab import *

import os.path

# Import solution
from solution import *

# Plot solution
figure(1)
plot(t, u[:, 0], t, u[:, 1], t, u[:, 2])
xlabel('t')
ylabel('U(t)')
title('Lorenz')


try :
  # import dual solution if it exists
  import solution_dual as dual
  figure(2)
  semilogy(dual.t, dual.u[:, 0], dual.t, dual.u[:, 1], dual.t, dual.u[:, 2])
  xlabel('t')
  ylabel('phi(t)')
  title('Lorenz (dual)')
except ImportError :
  pass

show()
