#!/bin/sh

VERSION=`../../../../config/dolfin-config --version`
DATE=`date`
CFLAGS=`../../../../config/dolfin-config --cflags`
TIMEFORMAT='real: %3R  user: %3U  sys: %3S  cpu: %P%%'

echo "DOLFIN version: $VERSION ($DATE)"
echo "Compiler flags: $CFLAGS"

echo -n "Elapsed time:   "
time ./dolfin-ode-bench > /dev/null
echo " "
uname -a
`../../../../config/dolfin-config --compiler` --version
echo "---------------------------------------------------------------------------"
