from Numeric import *
import MA
from pylab import *

#import primaldg as pcg
#import primalmdg as pmcg

import primalcg as pcg
import primalmcg as pmcg

t = pcg.t


udiff = pcg.u - pmcg.u
kdiff = pcg.k - pmcg.k
rdiff = pcg.r - pmcg.r

uexact = zeros(shape(pcg.u), 'd')
#uexact = sin(pcg.t)

#uexact[:, 0] = sin(pcg.t)
#uexact[:, 1] = cos(pcg.t)
#uexact[:, 2] = 0.0

uexact[:, 0] = 0.25 * pow(pcg.t, 4)
uexact[:, 1] = 0.0

e_cg = uexact - pcg.u
e_mcg = uexact - pmcg.u

print "e_cg:"
print MA.maximum(abs(e_cg))
print "e_mcg:"
print MA.maximum(abs(e_mcg))

plot(t, rdiff[:, 0])
