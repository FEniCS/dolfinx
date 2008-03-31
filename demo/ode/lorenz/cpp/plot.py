# Import matplotlib
from pylab import *

# Import solution
from solution import *

# Plot solution
figure(1)
plot(t, u[:, 0], t, u[:, 1], t, u[:, 2])
xlabel('t')
ylabel('U(t)')
title('Lorenz')

show()
