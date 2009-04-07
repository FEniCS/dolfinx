"""Run Valgrind on all demos."""

__author__ = "Ilmar Wilbers (ilmarw@simula.no)"
__date__ = "2008-04-08 -- 2008-06-22"
__copyright__ = "Copyright (C) 2008 Ilmar Wilbers"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg 2008
# Modified by Dag Lindbo 2008
# Modified by Johannes Ring 2008

import sys, os, re
import platform
from commands import getstatusoutput

if platform.system() in ['Darwin', 'Windows']:
    print "No support for Valgrind on this platform."
    sys.exit(0)

# Demos to run
cppdemos = []
for dpath, dnames, fnames in os.walk(os.path.join(os.curdir, "..", "..", "demo")):
    if os.path.basename(dpath) == 'cpp':
        if os.path.isfile(os.path.join(dpath, 'SConstruct')):
            cppdemos.append(dpath)
    
# Set non-interactive
os.putenv('DOLFIN_NOPLOT', '1')

# Helpful env vars
os.putenv('G_SLICE','always-malloc')
os.putenv('GLIBCXX_FORCE_NEW','1')
os.putenv('G_DEBUG','gc-friendly')

print "Running Valgrind on all demos (non-interactively)"
print ""
print "Found %d C++ demos" % len(cppdemos)
print ""

# Demos that need command line arguments are treated seperately
cppdemos.remove('./../../demo/quadrature/cpp')
cppdemos.remove('./../../demo/ode/method-weights/cpp')
cppdemos.remove('./../../demo/ode/stiff/cpp')

# Demos that are too time consuming to Valgrind
cppdemos.remove('./../../demo/pde/elastodynamics/cpp')
cppdemos.remove('./../../demo/pde/cahn-hilliard/cpp')

failed = []

dolfin_mpi_supp = os.path.join(os.getcwd(), 'dolfin_mpi.supp')
vg_comm = 'valgrind --error-exitcode=9 --tool=memcheck --leak-check=full --show-reachable=yes --suppressions=%s' % dolfin_mpi_supp

re_def_lost = re.compile("definitely lost: 0 bytes in 0 blocks.")
re_pos_lost = re.compile("possibly lost: 0 bytes in 0 blocks.")
re_reachable = re.compile("still reachable: 0 bytes in 0 blocks.")

# Run C++ demos
for demo in cppdemos:
    print "----------------------------------------------------------------------"
    print "Running Valgrind on C++ demo %s" % demo
    print ""
    if os.path.isfile(os.path.join(demo, 'demo')):
        output = getstatusoutput("cd %s && %s ./demo" % (demo, vg_comm) )
        noleak = re.findall('All heap blocks were freed',output[1])
        success = output[0] != 9 and len(noleak) == 1 and not "No such file or directory" in "".join([str(l) for l in output])
        if success:
            print "OK"
        else:
            failed += [(demo, "C++", output[1])]
            if output[0] == 9:
                print "*** FAILED: Memory error"
            elif output[1].find("LEAK SUMMARY:"):
                if re_def_lost.search(output[1]) and \
                   re_pos_lost.search(output[1]) and \
                   re_reachable.search(output[1]):
                    print "OK"
                else:
                    print "*** FAILED: Memory leak"
            elif len(noleak) != 1:
                print "*** FAILED: Memory leak"
            else:
                print "*** FAILED: Unable to run demo"
    else:
        print "*** Warning: missing demo"

# Print output for failed tests
print ""
if len(failed) > 0:
    print "%d demo(s) failed memcheck, see memcheck.log for details." % len(failed)
    file = open("memcheck.log", "w")
    for (test, interface, output) in failed:
        file.write("----------------------------------------------------------------------\n")
        file.write("%s (%s)\n" % (test, interface))
        file.write("\n")
        file.write(output)
        file.write("\n")
        file.write("\n")
else:
    print "All demos checked for memory leaks and errors: OK"

# Return error code if tests failed
sys.exit(len(failed) != 0)
