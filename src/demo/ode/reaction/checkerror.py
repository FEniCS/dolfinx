#!/usr/bin/python

from scipy.io.array_import import *

def checkerror(solution, reference):
    U = read_array(solution)
    u = read_array(reference)
    e = max(abs(U - u))
    return e

if __name__ == "__main__":

    e = checkerror("solution.data", "reference-gamma-100.data")
    print "Checking solution.data agains reference-gamma-100.data"
    print "Error in the maximum norm: %.3e" % e

    print ""

    e = checkerror("solution.data", "reference-gamma-1000.data")
    print "Checking solution.data agains reference-gamma-1000.data"
    print "Error in the maximum norm: %.3e" % e
