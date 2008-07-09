"""Run all demos."""

__author__ = "Ilmar Wilbers (ilmarw@simula.no)"
__date__ = "2008-04-08 -- 2008-04-09"
__copyright__ = "Copyright (C) 2008 Ilmar Wilbers"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg 2008

import sys, os, re
from commands import getstatusoutput
from time import time

# Demos to run
cppdemos = []
pydemos = []
for dpath, dnames, fnames in os.walk(os.path.join(os.curdir, "..", "..", "demo")):
    if os.path.basename(dpath) == 'cpp':
        if os.path.isfile(os.path.join(dpath, 'SConstruct')):
            cppdemos.append(dpath)
    elif os.path.basename(dpath) == 'python':
        if os.path.isfile(os.path.join(dpath, 'demo.py')):
            pydemos.append(dpath)
    
# Set non-interactive
os.putenv('DOLFIN_NOPLOT', '1')

print "Running all demos (non-interactively)"
print ""
print "Found %d C++ demos" % len(cppdemos)
print "Found %d Python demos" % len(pydemos)
print ""

# Remove demos that are known not to work (FIXME's)
pydemos.remove('./../../demo/nls/nonlinearpoisson/python')
pydemos.remove('./../../demo/pde/nonlinear-poisson/python')
pydemos.remove('./../../demo/pde/lift-drag/python')
pydemos.remove('./../../demo/ode/aliev-panfilov/python')

# Push slow demos to the end
pyslow = ['./../../demo/ode/lorenz/python']
cppslow = ['./../../demo/nls/cahn-hilliard/cpp']
for s in pyslow:
    pydemos.remove(s) 
    pydemos.append(s)
for s in cppslow:
    cppdemos.remove(s) 
    cppdemos.append(s)

# Demos that need command line arguments are treated seperately
pydemos.remove('./../../demo/quadrature/python')
cppdemos.remove('./../../demo/quadrature/cpp')
cppdemos.remove('./../../demo/ode/method-weights/cpp')
cppdemos.remove('./../../demo/ode/stiff/cpp')

failed = []
timing = []

# Run C++ demos
for demo in cppdemos:
    print "----------------------------------------------------------------------"
    print "Running C++ demo %s" % demo
    print ""
    if os.path.isfile(os.path.join(demo, 'demo')):
        t1 = time()
        output = getstatusoutput("cd %s && ./demo" % demo)
        t2 = time()
        timing += [(demo, t2-t1)]
        success = not output[0]
        if success:
            print "OK"
        else:
            print "*** Failed"
            failed += [(demo, "C++", output[1])]
    else:
        print "*** Warning: missing demo"

# Run Python demos
for demo in pydemos:
    print "----------------------------------------------------------------------"
    print "Running Python demo %s" % demo
    print ""
    if os.path.isfile(os.path.join(demo, 'demo.py')):
        t1 = time()
        output = getstatusoutput("cd %s && python ./demo.py" % demo)
        t2 = time()
        timing += [(demo, t2-t1)]
        success = not output[0]
        if success:
            print "OK"
        else:
            print "*** Failed"
            failed += [(demo, "Python", output[1])]
    else:
        print "*** Warning: missing demo"

# Print summary of time to run demos
if False:
    print ""
    print "Time to run demos:"
    print "\n".join("%s: %.2fs" % t for t in timing)

# Print output for failed tests
print ""
if len(failed) > 0:
    print "%d demo(s) failed, see demo.log for details." % len(failed)
    file = open("demo.log", "w")
    for (test, interface, output) in failed:
        file.write("----------------------------------------------------------------------\n")
        file.write("%s (%s)\n" % (test, interface))
        file.write("\n")
        file.write(output)
        file.write("\n")
        file.write("\n")
else:
    print "All demos checked: OK"

# Return error code if tests failed
sys.exit(len(failed) != 0)
