"Run all benchmarks"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2010-03-26 -- 2011-04-05"
__copyright__ = "Copyright (C) 2010 Anders Logg"
__license__  = "GNU LGPL version 2.1"

# Modified by Johannes Ring, 2011

import os, sys, time

failed = []

def run_bench(arg, directory, files):

    # Skip directories not containing a benchmark
    bench_exec = "bench"
    if directory.endswith("cpp") and \
           "bench_" + directory.split(os.path.sep)[-2] in files:
        bench_exec = "bench_" + directory.split(os.path.sep)[-2]


    if not bench_exec in files:
        return

    # Get name of benchmark
    name = directory.replace("./", "").replace("/", "-")
    print "Running benchmark %s..." % name

    # Remove old logfile
    cwd = os.getcwd()
    logfile = os.path.join(cwd, "logs", name + ".log")
    try:
        os.remove(logfile)
    except:
        pass

    # Run benchmark
    os.chdir(directory)
    t0 = time.time()
    status = os.system(os.path.join(os.curdir, bench_exec) + " > %s" % logfile)
    elapsed_time = time.time() - t0

    # Change to toplevel directory
    os.chdir(cwd)

    # Report timing
    if status == 0:
        print "Completed in %g seconds\n" % elapsed_time
    else:
        global failed
        failed.append(name)
        print "*** Failed\n"
        return

    # Get description of benchmark
    f = open(logfile)
    description = f.read().split("\n")[0]
    f.close()

    # Get timings (if any)
    f = open(logfile)
    timings = [("", elapsed_time)]
    for line in [line for line in f.read().split("\n") if "BENCH" in line]:
        words = [word.strip() for word in line.split(" ")]
        # Override total time
        if len(words) == 2:
            timings[0] = ("", float(words[1]))
        # Add sub timing
        elif len(words) == 3:
            timings.append((words[1].lower(), float(words[2])))
    f.close()

    # Append to log file
    d = time.gmtime()
    date = str((d.tm_year, d.tm_mon, d.tm_mday, d.tm_hour, d.tm_min, d.tm_sec))
    f = open(os.path.join("logs", "bench.log"), "a")
    for (postfix, timing) in timings:
        if postfix == "":
            n = name
            d = description
        else:
            n = "%s-%s" % (name, postfix)
            d = "%s (%s)" % (description, postfix)
        f.write('%s %s %g "%s"\n'  % (date, n, timing, d))

    return status == 0

# Iterate over benchmarks
os.path.walk(".", run_bench, None)

# Print summary
if len(failed) == 0:
    print "All benchmarks OK"
else:
    print "%d benchmark(s) failed:" % len(failed)
    for name in failed:
        print "  " + name

sys.exit(len(failed))
