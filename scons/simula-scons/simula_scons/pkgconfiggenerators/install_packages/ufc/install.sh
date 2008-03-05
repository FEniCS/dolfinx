#!/bin/bash

prefix=${1:-"/usr/local/ufc"}

echo "Install UFC 1.0 in $prefix"

builddir=$(mktemp -dt boostbuild)
cd $builddir

curl -O http://www.fenics.org/pub/software/ufc/v1.0/ufc-1.0.tar.gz
tar zxf ufc-1.0.tar.gz
cd ufc-1.0

cputype=$(uname -m)
arch=$(uname -s)

python setup.py install --prefix=$prefix


rm -rf $builddir
