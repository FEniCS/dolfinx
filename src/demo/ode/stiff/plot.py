# Import matplotlib
from pylab import *

#--- Test problem 1 ---
from solution_1 import *
figure(1)
plot(t, u)

#--- Test problem 2 ---
from solution_2 import *
figure(2)
plot(t, u[:,0], t, u[:,1])

#--- Test problem 3 ---
from solution_3 import *
figure(3)
plot(t, u[:,0], t, u[:,1])

#--- Test problem 4 ---
from solution_4 import *
figure(4)
plot(t, u[:,0], t, u[:,1], t, u[:,2], t, u[:,3], t, u[:,4], t, u[:,5], t, u[:,6], t, u[:,7])

#--- Test problem 5 ---
from solution_5 import *
figure(5)
plot(t, u[:,0], t, u[:,1], t, u[:,2], t, u[:,3], t, u[:,4], t, u[:,5])

#--- Test problem 6 ---
from solution_6 import *
figure(6)
plot(t, u[:,0], t, u[:,1])

#--- Test problem 7 ---
from solution_7 import *
figure(7)
plot(t, u[:,0], t, u[:,1], t, u[:,2], t, u[:,3], t, u[:,4], t, u[:,5], t, u[:,6], t, u[:,7], t, u[:,8], t, u[:,9])

#--- Test problem 8 ---
from solution_8 import *
figure(8)
plot(t, u[:,0], t, u[:,1], t, u[:,2])

#--- Test problem 9 ---
from solution_9 import *
figure(9)
plot(t, u[:,0], t, u[:,1], t, u[:,2])

show()
