"""Run Valgrind on all demos."""

# Copyright (C) 2008 Ilmar Wilbers
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg 2008
# Modified by Dag Lindbo 2008
# Modified by Johannes Ring 2008
# Modified by Johan Hake 2009
#
# First added:  2008-04-08
# Last changed: 2009-05-19

import sys, os, re
import platform
from commands import getstatusoutput

if "--only-python" in sys.argv:
    print "Skipping C++ only memory tests"
    sys.exit()

if platform.system() in ['Darwin', 'Windows']:
    print "No support for Valgrind on this platform."
    sys.exit(0)

# Demos to run
cppdemos = []
for dpath, dnames, fnames in os.walk(os.path.join(os.curdir, "..", "..", "demo")):
    if os.path.basename(dpath) == 'cpp':
        if os.path.isfile(os.path.join(dpath, 'Makefile')):
            cppdemos.append(dpath)

unit_test_excludes = ['graph']

# Python unit test to run
pythontests = []
for dpath, dnames, fnames in os.walk(os.path.join(os.curdir, "..", "unit")):
    if os.path.basename(dpath) == 'python':
        if os.path.isfile(os.path.join(dpath, 'test.py')):
            pythontests.append(dpath)

pythontests = [test for test in pythontests if not any([exclude in test for exclude in unit_test_excludes])]

unit_test_excludes.append('meshconvert')

# cpp unit test to run
cpptests = []
for dpath, dnames, fnames in os.walk(os.path.join(os.curdir, "..", "unit")):
    if os.path.basename(dpath) == 'cpp':
        if os.path.isfile(os.path.join(dpath, 'test')):
            cpptests.append(dpath)

cpptests = [test for test in cpptests if not any([exclude in test for exclude in unit_test_excludes])]

# Set non-interactive
os.putenv('DOLFIN_NOPLOT', '1')

# Helpful env vars
os.putenv('G_SLICE','always-malloc')
os.putenv('GLIBCXX_FORCE_NEW','1')
os.putenv('G_DEBUG','gc-friendly')

print pythontests

# Demos that need command line arguments are treated seperately
cppdemos.remove('./../../demo/undocumented/quadrature/cpp')
cppdemos.remove('./../../demo/undocumented/method-weights/cpp')
cppdemos.remove('./../../demo/undocumented/stiff/cpp')

# Demos that are too time consuming to Valgrind
cppdemos.remove('./../../demo/pde/cahn-hilliard/cpp')
cppdemos.remove('./../../demo/undocumented/elastodynamics/cpp')
cppdemos.remove('./../../demo/undocumented/reaction/cpp')
cppdemos.remove('./../../demo/undocumented/courtemanche/cpp')

re_def_lost = re.compile("definitely lost: 0 bytes in 0 blocks.")
re_pos_lost = re.compile("possibly lost: 0 bytes in 0 blocks.")
re_reachable = re.compile("still reachable: 0 bytes in 0 blocks.")
re_error = re.compile("0 errors from 0 contexts")

dolfin_supp = os.path.join(os.path.abspath(os.getcwd()), 'dolfin_valgrind.supp')
vg_comm = 'valgrind --error-exitcode=9 --tool=memcheck --leak-check=full --show-reachable=yes --suppressions=%s' % dolfin_supp

def run_and_analyse(path, run_str, prog_type, no_reachable_check = False):
    output = getstatusoutput("cd %s && %s %s" % (path, vg_comm, run_str))
    if "No such file or directory" in "".join([str(l) for l in output]):
        print "*** FAILED: Unable to run demo"
        return [(demo, "C++", output[1])]

    if len(re.findall('All heap blocks were freed',output[1])) == 1:
        print "OK"
        return []

    if "LEAK SUMMARY:" in output[1] and "ERROR SUMMARY:" in output[1]:
        if re_def_lost.search(output[1]) and \
           re_pos_lost.search(output[1]) and \
           re_error.search(output[1]) and \
           (no_reachable_check or re_reachable.search(output[1])):
            print "OK"
            return []

    print "*** FAILED: Memory error"
    return [(path, prog_type, output[1])]

failed = []

# Run C++ unittests
print "----------------------------------------------------------------------"
print "Running Valgrind on all C++ unittests"
print ""
print "Found %d C++ unittests" % len(cpptests)
print ""

for test_path in cpptests:
    print "----------------------------------------------------------------------"
    print "Running Valgrind on C++ unittest %s" % test_path
    print ""
    if os.path.isfile(os.path.join(test_path, 'test')):
        failed += run_and_analyse(test_path,"./test","C++")
    else:
        print "*** Warning: missing test"

# Outcommenting the Python unittests due to troubles with valgrind suppresions
#
#print "----------------------------------------------------------------------"
#print "Running Valgrind on all Python unittests"
#print ""
#print "Found %d Python unittests" % len(pythontests)
#print ""

# Run Python unittests
#for test_path in pythontests:
#    print "----------------------------------------------------------------------"
#    print "Running Valgrind on Python unittest %s" % test_path
#    print ""
#    if os.path.isfile(os.path.join(test_path, 'test.py')):
#        failed += run_and_analyse(test_path,"python test.py","Python",True)
#    else:
#        print "*** Warning: missing test"

# Run C++ demos
print "----------------------------------------------------------------------"
print "Running Valgrind on all demos (non-interactively)"
print ""
print "Found %d C++ demos" % len(cppdemos)
print ""

for demo_path in cppdemos:
    print "----------------------------------------------------------------------"
    print "Running Valgrind on C++ demo %s" % demo_path
    print ""
    demo_name = "./" + demo_path.split("/")[-2] + "-demo"
    print demo_name
    if os.path.isfile(os.path.join(demo_path, demo_name)):
        failed += run_and_analyse(demo_path, demo_name, "C++")
    else:
        print "*** Warning: missing demo"

# Print output for failed tests
print ""
if len(failed) > 0:
    print "%d demo(s) and/or unit test(s) failed memcheck, see memcheck.log for details." % len(failed)
    file = open("memcheck.log", "w")
    for (test, interface, output) in failed:
        file.write("----------------------------------------------------------------------\n")
        file.write("%s (%s)\n" % (test, interface))
        file.write("\n")
        file.write(output)
        file.write("\n")
        file.write("\n")
else:
    print "All demos and unit tests checked for memory leaks and errors: OK"

# Return error code if tests failed
sys.exit(len(failed) != 0)
