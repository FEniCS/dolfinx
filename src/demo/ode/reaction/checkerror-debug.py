#!/usr/bin/python

from Scientific.IO.ArrayIO import *

def checkerror(solution, reference):
    U = readFloatArray(solution)
    u = readFloatArray(reference)
    e = max(abs(U - u))
    return e

if __name__ == "__main__":

    e = checkerror("solution.data", "reference-gamma-1000-N-20.data")
    print "Checking solution.data agains reference-gamma-1000-N-20.data"
    print "Error in the maximum norm: %e" % e
