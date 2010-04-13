#!/usr/bin/env python

"""
This script parses logs/bench.log and create plots for each case with
the timings function of time (date plot). It also creates a web page
bench.html for easy viewing of the generated plots.
"""

__author__ = "Johannes Ring <johannr@simula.no>"
__date__ = "2010-04-06 -- 2010-04-13"
__copyright__ = "Copyright (C) 2010 Johannes Ring"
__license__  = "GNU LGPL version 2.1"

import os
import re
import time
import datetime
import textwrap
import numpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Change some of the default Matplotlib parameters
plt.rcParams.update({'figure.figsize': [6, 4],
                     'font.size' : 10,
                     'axes.labelsize' : 10,
                     'axes.grid': True,
                     'text.fontsize' : 10,
                     'legend.fontsize' : 8,
                     'xtick.labelsize' : 8,
                     'ytick.labelsize' : 8,
                     })

# Write to web page bench.html
outfile = open("bench.html", "w")
outfile.write("<h1><big>DOLFIN Benchmarks</big></h1>\n")
outfile.write("Last updated: %s.\n\n" % time.asctime())

# Open and read in logs/bench.log
benchlog = "logs/bench.log"
lines = open(benchlog, 'r').readlines()

benchmarks = {}
pattern = "\((.*)\)\s+(.*)\s+(\d+\.\d+|\d+)\s+\"(.*)\""

# Extract data from logfile
print "Parsing %s..." % benchlog
for line in lines:
    match = re.search(pattern, line)
    if match:
        year, month, day, hour, minute, second = \
              [int(i) for i in match.group(1).split(',')]
        #date = datetime.datetime(year, month, day, hour, minute, second)
        date = datetime.date(year, month, day)
        name = match.group(2)
        elapsed_time = float(match.group(3))
        description = match.group(4)

        if not name in benchmarks:
            benchmarks[name] = [[date], [elapsed_time], description]
        else:
            benchmarks[name][0].append(date)
            benchmarks[name][1].append(elapsed_time)

# Open and read in logs/milestones.log
milestones = []
milestoneslog = "logs/milestones.log"
if os.path.isfile(milestoneslog):
    lines = open(milestoneslog, 'r').readlines()
    for line in lines:
        date = datetime.datetime.strptime(line.split()[0], "%Y-%m-%d")
        progname = ' '.join(line.split()[1:])
        milestones.append([date, progname])

# Get Matplotlib line markers for use later
markers = []
for m in plt.Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

year = datetime.timedelta(days=365)
month = datetime.timedelta(days=30)
week = datetime.timedelta(days=7)
today = datetime.date.today()
lasts = ['week', 'month', 'year', 'five years']
locators = [mdates.DayLocator(), mdates.DayLocator(interval=2),
            mdates.MonthLocator(), mdates.YearLocator()]
date_fmts = ['%Y-%m-%d', '%d %b', '%b %Y', '%Y']
xmins = [today - week, today - month, today - year, today - 5*year]

outfile.write("<h2>All benchmarks</h2><p>\n")
outfile.write("<center>\n")
outfile.write("<table border=\"0\">\n")

# Create normalized plots with all benchmarks in same plot for
# last week, last month, last year, and last five years
print "Generating plots for all benchmarks..."
for last, locator, date_fmt, xmin in zip(lasts, locators, date_fmts, xmins):
    fig = plt.figure()
    ax = fig.gca()
    num = 0
    for benchmark, values in benchmarks.items():
        num += 1
        dates = values[0]
        timings = values[1]
        ax.plot(dates, timings/numpy.linalg.norm(timings),
                marker=markers[num % len(markers)], markersize=3,
                label=benchmark)
        ax.hold(True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.set_xlim(xmin, today)
    ax.set_ylim(0, 1)

    # Add milestones to plot
    for milestone in milestones:
        milestone_num = mdates.date2num(milestone[0])
        ax.annotate(milestone[1], xy=(milestone_num, 0.1E-10),
                    xycoords='data', xytext=(0, 30),
                    textcoords='offset points',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    style='italic', fontsize=6,
                    alpha=0.7, rotation='vertical',
                    arrowprops=dict(arrowstyle="->", alpha=0.3)
                    )

    lgd = plt.legend(loc='best')
    fig.autofmt_xdate()
    plt.title("All benchmarks (last %s)" % last)
    filename = "all_last_%s.png" % last.replace(' ', '_')
    plt.savefig(filename, facecolor='#eeeeee')

    # Add plots to web page
    if last in ['week', 'year']:
        outfile.write("  <tr><td><img src=\"%s\" /></td>\n" % filename)
    else:
        outfile.write("  <td><img src=\"%s\" /></td></tr>\n" % filename)

outfile.write("</table>\n")
outfile.write("</center>\n")

# Now create separate plots for every benchmark
for benchmark, values in benchmarks.items():
    print "Generating plots for %s..." % benchmark

    outfile.write("<h2>%s</h2><p>\n" % benchmark)
    outfile.write("<center>\n")
    outfile.write("<table border=\"0\">\n")

    dates = values[0]
    timings = values[1]
    description = values[2]
    # Wrap the lines in the description
    description = textwrap.fill(description, width=30)

    # Create plots for last week, last month, last year, and last five years
    for last, locator, date_fmt, xmin in zip(lasts, locators, date_fmts, xmins):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(dates, timings, marker='o', markersize=3)
        ax.set_ylabel("time (seconds)")
        ax.set_ylim(0, max(timings) + max(timings)/2.)
        ax.legend((description,), loc='best')        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        ax.set_xlim(xmin, today)

        # Add milestones to plot
        for milestone in milestones:
            milestone_num = mdates.date2num(milestone[0])
            ax.annotate(milestone[1], xy=(milestone_num, 0.1E-10),
                        xycoords='data', xytext=(0, 30),
                        textcoords='offset points',
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        style='italic', fontsize=6,
                        alpha=0.7, rotation='vertical',
                        arrowprops=dict(arrowstyle="->", alpha=0.3)
                        )

        fig.autofmt_xdate()
        plt.title("%s (last %s)" % (benchmark, last))
        filename = "%s_last_%s.png" % (benchmark, last.replace(' ', '_'))
        plt.savefig(filename, facecolor='#eeeeee')

        # Add plots to web page
        if last in ['week', 'year']:
            outfile.write("  <tr><td><img src=\"%s\" /></td>\n" % filename)
        else:
            outfile.write("  <td><img src=\"%s\" /></td></tr>\n" % filename)

    outfile.write("</table>\n")
    outfile.write("</center>\n")
