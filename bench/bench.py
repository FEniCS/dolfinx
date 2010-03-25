"Run all benchmarks"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2010-03-26 -- 2010-03-26"
__copyright__ = "Copyright (C) 2010 Anders Logg"
__license__  = "GNU LGPL version 2.1"

import os
import time

def run_bench(arg, directory, files):

    # Skip directories not containing a benchmark
    if not "bench" in files:
        return

    # Get name of benchmark
    name = directory.replace("./", "").replace("/", "-")

    # Run benchmark
    print "Running benchmark %s" % name

    # FIXME: Figure out how to present results

os.path.walk(".", run_bench, None)
