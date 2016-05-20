"""Run all demos."""

# Copyright (C) 2008-2014 Ilmar Wilbers
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
# Modified by Johannes Ring, 2009, 2011-2012.
# Modified by Johan Hake, 2009.
# Modified by Benjamin Kehlet 2012
# Modified by Martin Sandve Alnaes, 2014

from __future__ import print_function
import sys, os, re
import platform
import instant
from time import time
from instant import get_status_output
from dolfin import has_mpi, has_parmetis, has_scotch


def get_executable_name(demo, lang) :
  """Extract name of executable (without extension) from path.
     Name should be on the form demo_dir1_dir2 where dir1 and dir2 are
     directories under demo/[undocumented,documented] and lang (cpp or py)
     should be excluded.
  """
  directories = demo.split(os.path.sep)

  # Search for "demo" and lang from right
  directories_reverted = directories[::-1]
  demo_index = len(directories) - directories_reverted.index("demo") - 1
  lang_index = len(directories) - directories_reverted.index(lang) - 1
  truncated_directories = directories[demo_index+2:lang_index]
  return 'demo_' + "_".join(truncated_directories)

def run_cpp_demo(prefix, demo, rootdir, timing, failed):
    print("----------------------------------------------------------------------")
    print("Running C++ demo %s%s" % (prefix, demo))
    print("")

    cppdemo_executable = get_executable_name(demo, "cpp")
    if platform.system() == 'Windows':
        cppdemo_executable += '.exe'

    t1 = time()
    os.chdir(demo)
    status, output = get_status_output("%s .%s%s" % (prefix, os.path.sep, cppdemo_executable))
    os.chdir(rootdir)
    t2 = time()
    timing += [(t2 - t1, demo)]

    if status == 0:
        print("OK")
    elif status == 10: # Failing but exiting gracefully
        print("ok (graceful exit on fail)")
    else:
        print("*** Failed")
        print(output)
        failed += [(demo, "C++", prefix, output)]

def run_python_demo(prefix, demo, rootdir, timing, failed):
    print("----------------------------------------------------------------------")
    print("Running Python demo %s%s" % (prefix, demo))
    print("")

    demofile = get_executable_name(demo, "python") + '.py'

    t1 = time()
    os.chdir(demo)
    status, output = get_status_output("%s %s -u %s" % (prefix, sys.executable, demofile))
    os.chdir(rootdir)
    t2 = time()
    timing += [(t2 - t1, demo)]

    if status == 0:
        print("OK")
    elif status == 10: # Failing but exiting gracefully
        print("ok (graceful exit on fail)")
    else:
        print("*** Failed")
        print(output)

        # Add contents from Instant's compile.log to output
        instant_compile_log = os.path.join(instant.get_default_error_dir(), "compile.log")
        if os.path.isfile(instant_compile_log):
            instant_error = file(instant_compile_log).read()
            output += "\n\nInstant compile.log for %s:\n\n" % demo
            output += instant_error
        failed += [(demo, "Python", prefix, output)]

def main():
    # Location of all demos
    demodir = os.path.join(os.curdir, "..", "..", "demo")
    rootdir = os.path.abspath(os.curdir)

    # List of demos that have demo dir but are not currently implemented
    # NOTE: Demo must be listed here iff unimplemented otherwse the test will
    #       fail. This is meant to protect against usual bad named demos not
    #       executed for ages in regression tests.
    not_implemented = \
      [os.path.join(demodir, 'undocumented', 'projection-interpolation',    'cpp'), \
       os.path.join(demodir, 'undocumented', 'interpolation',               'cpp'), \
       os.path.join(demodir, 'undocumented', 'adaptive-poisson',            'cpp'), \
       os.path.join(demodir, 'undocumented', 'multistage-solver',           'cpp'), \
       os.path.join(demodir, 'undocumented', 'smoothing',                   'cpp'), \
       os.path.join(demodir, 'undocumented', 'overlapping-regions',         'cpp'), \
       os.path.join(demodir, 'undocumented', 'sub-function-assignment',     'cpp'), \
       os.path.join(demodir, 'undocumented', 'compiled-extension-module',   'cpp'), \
       os.path.join(demodir, 'undocumented', 'timing',                      'cpp'), \
       os.path.join(demodir, 'undocumented', 'coordinates',                 'cpp'), \
       os.path.join(demodir, 'documented',   'stokes-mini',                 'cpp'), \
       os.path.join(demodir, 'documented',   'tensor-weighted-poisson',     'cpp'), \
       os.path.join(demodir, 'documented',   'subdomains-poisson',          'cpp'), \
       ]

    # Demos to run
    cppdemos = []
    pydemos = []
    for dpath, dnames, fnames in os.walk(demodir):
        if os.path.basename(dpath) == 'cpp':
            if os.path.isfile(os.path.join(dpath, 'Makefile')):
                cppdemos.append(dpath)
                assert not dpath in not_implemented, \
                    "Demo '%s' marked as not_implemented" % dpath
            else:
                assert dpath in not_implemented, \
                    "Non-existing demo '%s' not marked as not_implemented" % dpath
        elif os.path.basename(dpath) == 'python':
            tmp = dpath.split(os.path.sep)[-2]
            if os.path.isfile(os.path.join(dpath, 'demo_' + tmp + '.py')):
                pydemos.append(dpath)
                assert not dpath in not_implemented, \
                    "Demo '%s' marked as not_implemented" % dpath
            else:
                assert dpath in not_implemented, \
                    "Non-existing demo '%s' not marked as not_implemented" % dpath

    # Set non-interactive
    os.putenv('DOLFIN_NOPLOT', '1')

    print("Running all demos (non-interactively)")
    print("")
    print("Found %d C++ demos" % len(cppdemos))
    print("Found %d Python demos" % len(pydemos))
    print("")
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

    # List of demos that throw expected errors in parallel
    not_working_in_parallel = \
      [os.path.join(demodir, 'undocumented', 'adaptive-poisson',            'python'),
       os.path.join(demodir, 'undocumented', 'auto-adaptive-navier-stokes', 'cpp'),
       os.path.join(demodir, 'undocumented', 'auto-adaptive-navier-stokes', 'python'),
       os.path.join(demodir, 'documented',   'auto-adaptive-poisson',       'cpp'),
       os.path.join(demodir, 'documented',   'auto-adaptive-poisson',       'python'),
       os.path.join(demodir, 'undocumented', 'eval',                        'cpp'),
       os.path.join(demodir, 'undocumented', 'eval',                        'python'),
       os.path.join(demodir, 'undocumented', 'extrapolation',               'cpp'),
       os.path.join(demodir, 'undocumented', 'extrapolation',               'python'),
       os.path.join(demodir, 'undocumented', 'meshfunction-refinement',     'cpp'),
       os.path.join(demodir, 'undocumented', 'nonmatching-interpolation',   'cpp'),
       os.path.join(demodir, 'undocumented', 'nonmatching-interpolation',   'python'),
       os.path.join(demodir, 'undocumented', 'nonmatching-projection',      'cpp'),
       os.path.join(demodir, 'undocumented', 'nonmatching-projection',      'python'),
       os.path.join(demodir, 'undocumented', 'poisson-disc',                'python'),
       os.path.join(demodir, 'undocumented', 'poisson-disc',                'cpp'),
       os.path.join(demodir, 'undocumented', 'smoothing',                   'python'),
       os.path.join(demodir, 'documented',   'subdomains',                  'cpp'),
       os.path.join(demodir, 'documented',   'subdomains',                  'python'),
       os.path.join(demodir, 'undocumented', 'submesh',                     'cpp'),
       os.path.join(demodir, 'undocumented', 'submesh',                     'python'),
       os.path.join(demodir, 'undocumented', 'time-series',                 'cpp'),
       os.path.join(demodir, 'undocumented', 'time-series',                 'python'),
       os.path.join(demodir, 'undocumented', 'triangulate',                 'cpp'),
       os.path.join(demodir, 'undocumented', 'triangulate',                 'python'),
       os.path.join(demodir, 'undocumented', 'poisson1D-in-2D',             'cpp'),
       os.path.join(demodir, 'undocumented', 'poisson1D-in-2D',             'python'),
       os.path.join(demodir, 'undocumented', 'compiled-extension-module',   'python'),
       os.path.join(demodir, 'undocumented', 'coordinates',                 'python'),
       ]

    failed = []
    timing = []

    # Check if we should run only Python tests, use for quick testing
    if len(sys.argv) == 2 and sys.argv[1] == "--only-python":
        only_python = True
    else:
        only_python = False

    # Check if we should skip C++ demos
    if only_python:
        print("Skipping C++ demos")
        cppdemos = []

    # Build prefix list
    prefixes = [""]
    mpi_prefix = "mpirun -np 3 "
    if has_mpi() and (has_parmetis() or has_scotch()):
        prefixes.append(mpi_prefix)
    else:
        print("Not running regression tests in parallel.")

    # Allow to disable parallel testing
    if "DISABLE_PARALLEL_TESTING" in os.environ:
        prefixes = [""]

    # Run in serial, then in parallel
    for prefix in prefixes:

        # List of demos to run
        if prefix == mpi_prefix:
            cppdemos_to_run = list(set(cppdemos) - set(not_working_in_parallel))
            pydemos_to_run  = list(set(pydemos)  - set(not_working_in_parallel))
        else:
            cppdemos_to_run = cppdemos
            pydemos_to_run  = pydemos

        # Run demos
        for demo in cppdemos_to_run:
            run_cpp_demo(prefix, demo, rootdir, timing, failed)
        for demo in pydemos_to_run:
            run_python_demo(prefix, demo, rootdir, timing, failed)

    # Print summary of time to run demos
    timing.sort()
    print("")
    print("Time to run demos:")
    print("\n".join("%.2fs: %s" % t for t in timing))

    total_no_demos = len(pydemos)
    if not only_python:
        total_no_demos += len(cppdemos)

    # Print output for failed tests
    print("")
    if len(failed) > 0:
        print("%d demo(s) out of %d failed, see demo.log for details." %
              (len(failed), total_no_demos))
        file = open("demo.log", "w")
        for (test, interface, prefix, output) in failed:
            file.write("----------------------------------------------------------------------\n")
            file.write("%s%s (%s)\n" % (prefix, test, interface))
            file.write("\n")
            file.write(output)
            file.write("\n")
            file.write("\n")
    else:
        print("All demos checked: OK")

    # Return error code if tests failed
    return len(failed)

if __name__ == "__main__":
    sys.exit(main())
