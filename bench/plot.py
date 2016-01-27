#!/usr/bin/env python

"""
This script parses logs/bench.log and create plots for each case with
the timings function of time (date plot). It also creates a web page
index.html for easy viewing of the generated plots.
"""

# Copyright (C) 2010 Johannes Ring
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
# First added:  2010-04-06
# Last changed: 2010-04-13

from __future__ import print_function
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

# Write to web page index.html
outfile = open("index.html", "w")
outfile.write("<h1><big>DOLFIN Benchmarks</big></h1>\n")
outfile.write("Last updated: %s.\n\n" % time.asctime())

# Open and read in logs/bench.log
benchlog = "logs/bench.log"
lines = open(benchlog, 'r').readlines()

benchmarks = {}
pattern = "\((.*)\)\s+(.*)\s+(.*)\s+\"(.*)\""

# Extract data from logfile
print("Parsing %s..." % benchlog)
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
            benchmarks[name] = [[[date], [elapsed_time], description]]
        else:
            benchmarks[name][0][0].append(date)
            benchmarks[name][0][1].append(elapsed_time)
            benchmarks[name][0][2] = description # take later description

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

def get_maxtime(dates, min_date, max_date, run_timings):
    """Return the maximum time between min_date and max_date"""
    max_time = 0
    for i, date in enumerate(dates):
        if date < min_date:
            continue
        elif date > max_date:
            break
        else:
            if max_time < run_timings[i]:
                max_time = run_timings[i]
    return max_time

# Create normalized plots with all benchmarks in same plot for
# last week, last month, last year, and last five years
linetypes = {}
print("Generating plots for all benchmarks...")
for last, locator, date_fmt, xmin in zip(lasts, locators, date_fmts, xmins):
    fig = plt.figure()
    ax = fig.gca()
    num = 0
    ymax = 0
    for benchmark, values in benchmarks.items():
        num += 1
        dates = values[0][0]
        run_timings = values[0][1]/numpy.linalg.norm(values[0][1])
        p = ax.plot(dates, run_timings,
                    marker=markers[num % len(markers)], markersize=3,
                    label=benchmark)
        ax.hold(True)
        linetypes[benchmark] = p[0]
        maxtime = get_maxtime(dates, xmin, today, run_timings)
        if maxtime > ymax:
            ymax = maxtime
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.set_xlim(xmin, today)
    ax.set_ylim(0, ymax)

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

    #lgd = plt.legend(loc='best')
    fig.autofmt_xdate()
    plt.title("All benchmarks (last %s)" % last)
    filename = "all_last_%s.png" % last.replace(' ', '_')
    plt.savefig(filename, facecolor='#eeeeee')

    # Add plots to web page
    if last in ['week', 'year']:
        outfile.write("  <tr><td><img src=\"%s\" /></td>\n" % filename)
    else:
        outfile.write("  <td><img src=\"%s\" /></td></tr>\n" % filename)

# Separate legend
linetypes = sorted(linetypes.items())
handles = [p for (t, p) in linetypes]
labels = [t for (t, p) in linetypes]
plt.figure(figsize=(12, 4)) # twice wide the default
plt.figlegend(handles, labels, 'center', ncol=3)
plt.savefig('all_legend.png')
outfile.write("  <tr><td colspan=\"2\"><img src=\"%s\" /></td></tr>\n" % 'all_legend.png')

outfile.write("</table>\n")
outfile.write("</center>\n")

# Put together subtasks benchmarks which should be plotted together
# (annotated by ((...)) in description)
for benchmark, values in benchmarks.items():
    postfix = values[0][2].split("((")
    if len(postfix) == 1:
        continue
    assert len(postfix) == 2
    postfix = postfix[1].strip().replace("))", "")
    benchmarks.pop(benchmark)
    name = "-".join(benchmark.split("-")[:3]) + "-subtasks"
    if not name in benchmarks:
        benchmarks[name] = []
    benchmarks[name].append((values[0][0], values[0][1], postfix))

# Now create separate plots for every benchmark
for benchmark, values in sorted(benchmarks.items()):
    print("Generating plots for %s..." % benchmark)

    outfile.write("<h2>%s</h2><p>\n" % benchmark)
    outfile.write("<center>\n")
    outfile.write("<table border=\"0\">\n")

    # Create plots for last week, last month, last year, and last five years
    for last, locator, date_fmt, xmin in zip(lasts, locators, date_fmts, xmins):
        fig = plt.figure()
        ax = fig.gca()
        if any(s in benchmark for s in ["fem-speedup-cpp-"]):
            ax.set_ylabel("speedup")
        else:
            ax.set_ylabel("time (seconds)")
        maxtime = 0.0
        for v in values:
            dates = v[0]
            run_timings = v[1]
            description = textwrap.fill(v[2], width=30)
            maxtime = max(maxtime, get_maxtime(dates, xmin, today, run_timings))
            ax.set_ylim(0, maxtime + maxtime/2)
            ax.plot(dates, run_timings, marker='o', markersize=3, label=description)
        ax.legend(loc='best')
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
