#!/bin/bash

prefix=${1:-"/usr/local/cmake"}

echo "Install cmake 2.4 in $prefix"

builddir=$(mktemp -dt boostbuild)
cd $builddir

curl -O http://www.cmake.org/files/v2.4/cmake-2.4.7.tar.gz
tar zxf cmake-2.4.7.tar.gz
cd cmake-2.4.7

cputype=$(uname -m)
arch=$(uname -s)


./configure --prefix=$prefix
make
make install


rm -rf $builddir
