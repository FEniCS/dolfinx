#!/bin/bash

prefix=${1:-"/usr/local/ufc"}

echo "Install pkg config 0.22 in $prefix"

builddir=$(mktemp -dt boostbuild)
cd $builddir

curl -O http://pkgconfig.freedesktop.org/releases/pkg-config-0.22.tar.gz
tar zxf pkg-config-0.22.tar.gz
cd pkg-config-0.22

cputype=$(uname -m)
arch=$(uname -s)


./configure --prefix=$prefix
make
make install


rm -rf $builddir
