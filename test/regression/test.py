"""Run all demos."""

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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008-2009.
# Modified by Johannes Ring, 2009, 2011.
# Modified by Johan Hake, 2009.
#
# First added:  2008-04-08
# Last changed: 2011-06-23

import sys, os, re
import platform
import instant
from time import time
from dolfin_utils.commands import getstatusoutput
from dolfin import has_mpi, has_parmetis

# Location of all demos
demodir = os.path.join(os.curdir, "..", "..", "demo")

# Demos to run
cppdemos = []
pydemos = []
for dpath, dnames, fnames in os.walk(demodir):
    if os.path.basename(dpath) == 'cpp':
        if os.path.isfile(os.path.join(dpath, 'Makefile')):
            cppdemos.append(dpath)
    elif os.path.basename(dpath) == 'python':
        tmp = dpath.split(os.path.sep)[-2]
        if os.path.isfile(os.path.join(dpath, 'demo_' + tmp + '.py')):
            pydemos.append(dpath)

# Set non-interactive
os.putenv('DOLFIN_NOPLOT', '1')

print "Running all demos (non-interactively)"
print ""
print "Found %d C++ demos" % len(cppdemos)
print "Found %d Python demos" % len(pydemos)
print ""
import pprint

# Push slow demos to the end
pyslow = []
cppslow = []
for s in pyslow:
    if s in pydemos:
        pydemos.remove(s)
        pydemos.append(s)
for s in cppslow:
    if s in cppdemos:
        cppdemos.remove(s)
        cppdemos.append(s)

# Remove demos that need command-line arguments
pyremoves  = [os.path.join(demodir,  'undocumented', 'quadrature', 'python')]
cppremoves = [os.path.join(demodir,  'undocumented', 'quadrature', 'cpp')]
for demo in pyremoves:
    if demo in pydemos:
        pydemos.remove(demo)

for demo in cppremoves:
    if demo in cppdemos:
        cppdemos.remove(demo)

# Remove C++ coloring demo on Windows until #797640 is fixed
if platform.system() == 'Windows':
    winremove = [os.path.join(demodir, 'undocumented', 'coloring', 'cpp')]
    for demo in wrinremove:
        if demo in cppdemos:
            cppdemos.remove()

# List of demos that throw expected errors in parallel
not_working_in_parallel = \
  [os.path.join(demodir, 'pde',          'biharmonic',                  'cpp'),    \
   os.path.join(demodir, 'pde',          'biharmonic',                  'python'), \
   os.path.join(demodir, 'undocumented', 'advection-difusion',          'cpp'),    \
   os.path.join(demodir, 'undocumented', 'advection-difusion',          'python'), \
   os.path.join(demodir, 'undocumented', 'ale',                         'cpp'),    \
   os.path.join(demodir, 'undocumented', 'ale',                         'python'), \
   os.path.join(demodir, 'undocumented', 'auto-adaptive-navier-stokes', 'cpp'),    \
   os.path.join(demodir, 'undocumented', 'auto-adaptive-navier-stokes', 'python'), \
   os.path.join(demodir, 'undocumented', 'auto-adaptive-poisson',       'cpp'),    \
   os.path.join(demodir, 'undocumented', 'auto-adaptive-poisson',       'python'), \
   os.path.join(demodir, 'undocumented', 'dg-advection-diffusion',      'cpp'),    \
   os.path.join(demodir, 'undocumented', 'dg-advection-diffusion',      'python'), \
   os.path.join(demodir, 'undocumented', 'dg-poisson',                  'cpp'),    \
   os.path.join(demodir, 'undocumented', 'dg-poisson',                  'python'), \
   os.path.join(demodir, 'undocumented', 'eval',                        'cpp'),    \
   os.path.join(demodir, 'undocumented', 'eval',                        'python'), \
   os.path.join(demodir, 'undocumented', 'meshfunction',                'cpp'),    \
   os.path.join(demodir, 'undocumented', 'meshfunction',                'python'), \
   os.path.join(demodir, 'undocumented', 'stokes-stabilized',           'cpp'),    \
   os.path.join(demodir, 'undocumented', 'stokes-stabilized',           'python'), \
   os.path.join(demodir, 'undocumented', 'advection-diffusion',         'cpp'),    \
   os.path.join(demodir, 'undocumented', 'advection-diffusion',         'python'), \
   os.path.join(demodir, 'undocumented', 'refinement',                  'cpp'),    \
   os.path.join(demodir, 'undocumented', 'refinement',                  'python'), \
   os.path.join(demodir, 'undocumented', 'extrapolation',               'cpp'),    \
   os.path.join(demodir, 'undocumented', 'extrapolation',               'python'), \
   os.path.join(demodir, 'undocumented', 'nonmatching-interpolation',   'cpp'),    \
   os.path.join(demodir, 'undocumented', 'nonmatching-interpolation',   'python'), \
   os.path.join(demodir, 'undocumented', 'time-series',                 'cpp'),    \
   os.path.join(demodir, 'undocumented', 'time-series',                 'python'), \
   os.path.join(demodir, 'undocumented', 'subdomains',                  'cpp'),    \
   os.path.join(demodir, 'undocumented', 'subdomains',                  'python'), \
   os.path.join(demodir, 'undocumented', 'stokes-taylor-hood',          'cpp'),    \
   os.path.join(demodir, 'undocumented', 'stokes-taylor-hood',          'python'), \
   os.path.join(demodir, 'undocumented', 'meshfunction-refinement',     'cpp'),    \
   os.path.join(demodir, 'undocumented', 'meshfunction-refinement',     'python'), \
   os.path.join(demodir, 'undocumented', 'neumann-possion',             'cpp'),    \
   os.path.join(demodir, 'undocumented', 'neumann-possion',             'python'), \
   os.path.join(demodir, 'undocumented', 'nonmatching-projection',      'cpp'),    \
   os.path.join(demodir, 'undocumented', 'nonmatching-projection',      'python'), \
   os.path.join(demodir, 'undocumented', 'partitioning',                'cpp'),    \
   os.path.join(demodir, 'undocumented', 'partitioning',                'python'), \
   os.path.join(demodir, 'undocumented', 'simple',                      'cpp'),    \
   os.path.join(demodir, 'undocumented', 'simple'      ,                'python'), \
   os.path.join(demodir, 'undocumented', 'periodic',                    'cpp'),    \
   os.path.join(demodir, 'undocumented', 'periodic',                    'python'), \
   os.path.join(demodir, 'undocumented', 'adaptive-poisson',            'cpp'),    \
   os.path.join(demodir, 'undocumented', 'adaptive-poisson',            'python')]


failed = []
timing = []

# Check if we should run only Python tests, use for quick testing
if len(sys.argv) == 2 and sys.argv[1] == "--only-python":
    only_python = True
else:
    only_python = False

# Check if we should skip C++ demos
if only_python:
    print "Skipping C++ demos"
    cppdemos = []

# Build prefix list
prefixes = [""]
mpi_prefix = "mpirun -np 3 "
if "RUN_REGRESSION_TESTS_IN_PARALLEL" in os.environ and has_mpi() and has_parmetis():
    prefixes.append(mpi_prefix)
else:
    print "Not running regression tests in parallel."

# Run in serial, then in parallel
for prefix in prefixes:

    # List of demos to run
    if prefix == mpi_prefix:
        cppdemos_to_run = list(set(cppdemos) - set(not_working_in_parallel))
        pydemos_to_run  = list(set(pydemos)  - set(not_working_in_parallel))
    else:
        cppdemos_to_run = cppdemos
        pydemos_to_run  = pydemos

    # Run C++ demos
    for demo in cppdemos_to_run:
        print "----------------------------------------------------------------------"
        print "Running C++ demo %s%s" % (prefix, demo)
        print ""
        cppdemo_executable = 'demo_' + demo.split(os.path.sep)[-2]
        if platform.system() == 'Windows':
            cppdemo_executable += '.exe'
        if os.path.isfile(os.path.join(demo, cppdemo_executable)):
            t1 = time()
            output = getstatusoutput("cd %s && %s .%s%s" % \
                                     (demo, prefix, os.path.sep, cppdemo_executable))
            t2 = time()
            timing += [(t2 - t1, demo)]
            if output[0] == 0:
                print "OK"
            elif output[0] == 10: # Failing but exiting gracefully
                print "ok (graceful exit on fail)"
            else:
                print "*** Failed"
                print output[1]
                failed += [(demo, "C++", prefix, output[1])]
        else:
            print "*** Warning: missing demo"

    # Run Python demos
    for demo in pydemos_to_run:
        print "----------------------------------------------------------------------"
        print "Running Python demo %s%s" % (prefix, demo)
        print ""
        demofile = 'demo_' + demo.split(os.path.sep)[-2] + '.py'
        if os.path.isfile(os.path.join(demo, demofile)):
            t1 = time()
            status, output = getstatusoutput("cd %s && %s python %s" % (demo, prefix, demofile))
            t2 = time()
            timing += [(t2 - t1, demo)]
            if status == 0:
                print "OK"
            elif status == 10: # Failing but exiting gracefully
                print "ok (graceful exit on fail)"
            else:
                print "*** Failed"
                print output

                # Add contents from Instant's compile.log to output
                instant_compile_log = os.path.join(instant.get_default_error_dir(), "compile.log")
                if os.path.isfile(instant_compile_log):
                    instant_error = file(instant_compile_log).read()
                    output += "\n\nInstant compile.log for %s:\n\n" % demo
                    output += instant_error
                failed += [(demo, "Python", prefix, output)]
        else:
            print "*** Warning: missing demo"

# Print summary of time to run demos
timing.sort()
print ""
print "Time to run demos:"
print "\n".join("%.2fs: %s" % t for t in timing)

total_no_demos = len(pydemos)
if not only_python:
    total_no_demos += len(cppdemos)

# Print output for failed tests
print ""
if len(failed) > 0:
    print "%d demo(s) out of %d failed, see demo.log for details." % \
          (len(failed), total_no_demos)
    file = open("demo.log", "w")
    for (test, interface, prefix, output) in failed:
        file.write("----------------------------------------------------------------------\n")
        file.write("%s%s (%s)\n" % (prefix, test, interface))
        file.write("\n")
        file.write(output)
        file.write("\n")
        file.write("\n")
else:
    print "All demos checked: OK"

# Return error code if tests failed
sys.exit(len(failed) != 0)
