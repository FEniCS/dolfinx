# Import solution
from solution import *
import stability_factors

# Import matplotlib
from pylab import *

# Plot solution
figure(1)
plot(t, u[:, 0], t, u[:, 1], t, u[:, 2])
xlabel('t')
ylabel('U(t)')
title('Lorenz')

# Plot stability factors
figure(2)
semilogy(stability_factors.t, stability_factors.u[:, 0], 
         stability_factors.t, stability_factors.u[:, 1], 
         stability_factors.t, stability_factors.u[:, 2])
xlabel('$T$')
ylabel('$S_C$')
title('Computational stability factors of the Lorenz system')

show()
