#!/bin/bash

prefix=${1:-"/usr/local/dolfin"}

echo "Install DOLFIN from hg in $prefix"

builddir=$(mktemp -dt boostbuild)
cd $builddir

hg clone http://www.fenics.org/hg/dolfin
cd dolfin

cputype=$(uname -m)
arch=$(uname -s)


./configure --prefix=$prefix
make
make install

rm -rf $builddir
