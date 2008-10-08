# Import solutions
import solution_1 as s1
import solution_2 as s2
import solution_3 as s3
import solution_4 as s4
import solution_5 as s5
import solution_6 as s6
import solution_7 as s7
import solution_8 as s8
import solution_9 as s9

# Import matplotlib
from pylab import *

#--- Test problem 1 ---
figure(1)
plot(s1.t, s1.u)

#--- Test problem 2 ---
figure(2)
plot(s2.t, s2.u[:,0], s2.t, s2.u[:,1])

#--- Test problem 3 ---
figure(3)
plot(s3.t, s3.u[:,0], s3.t, s3.u[:,1])

#--- Test problem 4 ---
figure(4)
plot(s4.t, s4.u[:,0], s4.t, s4.u[:,1], s4.t, s4.u[:,2], s4.t, s4.u[:,3], s4.t, s4.u[:,4], s4.t, s4.u[:,5], s4.t, s4.u[:,6], s4.t, s4.u[:,7])

#--- Test problem 5 ---
#figure(5)
#plot(s5.t, s5.u[:,0], s5.t, s5.u[:,1], s5.t, s5.u[:,2], s5.t, s5.u[:,3], s5.t, s5.u[:,4], s5.t, s5.u[:,5])

#--- Test problem 6 ---
figure(6)
plot(s6.t, s6.u[:,0], s6.t, s6.u[:,1])

#--- Test problem 7 ---
figure(7)
plot(s7.t, s7.u[:,0], s7.t, s7.u[:,1], s7.t, s7.u[:,2], s7.t, s7.u[:,3], s7.t, s7.u[:,4], s7.t, s7.u[:,5], s7.t, s7.u[:,6], s7.t, s7.u[:,7], s7.t, s7.u[:,8], s7.t, s7.u[:,9])

#--- Test problem 8 ---
figure(8)
plot(s8.t, s8.u[:,0], s8.t, s8.u[:,1], s8.t, s8.u[:,2])

#--- Test problem 9 ---
figure(9)
plot(s9.t, s9.u[:,0], s9.t, s9.u[:,1], s9.t, s9.u[:,2])

show()
