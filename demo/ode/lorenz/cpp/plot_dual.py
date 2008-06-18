#!/usr/bin/env python

__author__ = "Benjamin Kehlet <benjamik@ifi.uio.no>"
__date__ = "2008-06-11 -- 2008-06-18"
__copyright__ = "Copyright (C) 2008 Benjamin Kehlet"
__license__  = "GNU LGPL Version 2.1"

# Import matplotlib
from pylab import *

# Import solution
from solution_dual import *

# Plot solution
figure(1)
semilogy(t, u[:, 0], t, u[:, 1], t, u[:, 2])
xlabel('t')
ylabel('phi(t)')
title('Lorenz (dual)')

show()
