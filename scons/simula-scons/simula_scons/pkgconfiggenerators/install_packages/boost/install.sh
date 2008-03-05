#!/bin/bash

prefix=${1:-"/usr/local/boost"}

echo "Install Boost 1.34.1 in $prefix"

builddir=$(mktemp -dt boostbuild)
cd $builddir

curl -O http://kent.dl.sourceforge.net/sourceforge/boost/boost_1_34_1.tar.gz
tar zxf boost_1_34_1.tar.gz
cd boost_1_34_1

cputype=$(uname -m)
arch=$(uname -s)

./configure --prefix=$prefix
make
make install



# finally, remove the builddir

rm -rf $builddir
