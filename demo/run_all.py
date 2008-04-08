"""Run all demos."""

__author__ = "Ilmar Wilbers (ilmarw@simula.no)"
__date__ = "2008-04-04"
__copyright__ = "Copyright (C) 2006-2007 Ilmar Wilbers"
__license__  = "GNU LGPL Version 2.1"

import sys, os, re
from commands import getstatusoutput

# Demos to run
cppdemos = []
pydemos = []

for dpath, dnames, fnames in os.walk(os.curdir):
    if os.path.basename(dpath) == 'cpp':
        if os.path.isfile(os.path.join(dpath, 'demo')):
            cppdemos.append(dpath)
    elif os.path.basename(dpath) == 'python':
        pydemos.append(dpath)
    
# Set non-interactive:
os.putenv('DOLFIN_NOPLOT', '1')

print "This script is used for testing purposes,"
print "all demos are run non-interactive\n"

# Remove demos that are known not to work (FIXME's):
pydemos.remove('./nls/nonlinearpoisson/python')
pydemos.remove('./pde/nonlinear-poisson/python')
pydemos.remove('./pde/lift-drag/python')
pydemos.remove('./ode/aliev-panfilov/python')

# Demos that need command line arguments are treated seperately:
pydemos.remove('./quadrature/python')
cppdemos.remove('./quadrature/cpp')
cppdemos.remove('./ode/method-weights/cpp')
cppdemos.remove('./ode/stiff/cpp')

# Run python demos
failed = []
for demo in pydemos:
    if os.path.isfile(os.path.join(demo, 'demo.py')):
        print "Running Python demo for %s" % demo
        print "----------------------------------------------------------------------"
        output = getstatusoutput("cd %s && python ./demo.py" % demo)
        success = not output[0]
        if success:
            print "OK"
        else:
            print "*** Failed"
            failed += [(demo, "Python", output[1])]

        print ""

for demo in cppdemos:  
    if os.path.isfile(os.path.join(demo, 'demo')):
        print "Running C++ demo for %s" % demo
        print "----------------------------------------------------------------------" 
        output = getstatusoutput("cd %s && ./demo" % demo)
        success = not output[0]
        if success:
            print "OK"
        else:
            print "*** Failed"
            failed += [(demo, "C++", output[1])]

        print ""

# Print output for failed tests
for (test, interface, output) in failed:
    print "One or more demos failed for %s (%s):" % (test, interface)
    print output

# Return error code if tests failed
sys.exit(len(failed) != 0)
