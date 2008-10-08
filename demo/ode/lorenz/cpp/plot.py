# Import solution
from solution import *
import solution_dual as dual

# Import matplotlib
from pylab import *

# Plot solution
figure(1)
plot(t, u[:, 0], t, u[:, 1], t, u[:, 2])
xlabel('t')
ylabel('U(t)')
title('Lorenz')

# Plot dual solution
figure(2)
semilogy(dual.t, abs(dual.u[:, 0]), dual.t, abs(dual.u[:, 1]), dual.t, abs(dual.u[:, 2]))
xlabel('t')
ylabel('phi(t)')
title('Lorenz (dual)')

show()
