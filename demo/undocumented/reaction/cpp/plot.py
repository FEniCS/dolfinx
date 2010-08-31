# Import solution
from solution import *

# Import matplotlib
from pylab import *

# Plot solution
figure(1)
plot(t, u)
grid(True)
xlabel('t')
ylabel('u(t)')
title('Solution')

# Plot solution
figure(2)
plot(t, k)
grid(True)
xlabel('t')
ylabel('k(t)')
title('Multi-adaptive time steps')

show()
