"Run all benchmarks"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2010-03-26 -- 2010-03-26"
__copyright__ = "Copyright (C) 2010 Anders Logg"
__license__  = "GNU LGPL version 2.1"

import os
import time

failed = []

def run_bench(arg, directory, files):

    # Skip directories not containing a benchmark
    if not "bench" in files:
        return

    # Get name of benchmark
    name = directory.replace("./", "").replace("/", "-")
    print "Running benchmark %s..." % name

    # Remove old logfile
    cwd = os.getcwd()
    logfile = os.path.join(cwd, "log", name + ".log")
    try:
        os.remove(logfile)
    except:
        pass

    # Run benchmark
    os.chdir(directory)
    t0 = time.time()
    status = os.system("./bench" + " > %s" % logfile)
    elapsed_time = time.time() - t0

    # Report timing
    if status == 0:
        print "Completed in %g seconds\n" % elapsed_time
    else:
        print "*** Failed\n"

    # Change to toplevel directory
    os.chdir(cwd)

    # Append to log file
    d = time.gmtime()
    date = str((d.tm_year, d.tm_mon, d.tm_mday, d.tm_hour, d.tm_min, d.tm_sec))
    f = open(os.path.join("log", "bench.log"), "a")
    f.write("%s %s %g\n" % (date, name, elapsed_time))
    f.close()

    return status == 0

os.path.walk(".", run_bench, None)
