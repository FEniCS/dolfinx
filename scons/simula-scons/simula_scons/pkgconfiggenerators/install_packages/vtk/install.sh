#!/bin/bash

prefix=${1:-"/usr/local/vtk"}

echo "Install VTK 5.0.3 in $prefix"

builddir=$(mktemp -d /tmp/vtk.XXXX)
cd $builddir

curl -O -k http://www.vtk.org/files/release/5.0/vtk-5.0.3.tar.gz
tar xzf vtk-5.0.3.tar.gz
cd VTK

cputype=$(uname -m)
arch=$(uname -s)

if [ $cputype == "i386" -a $arch == "Darwin" ] ; then
    echo "Configuring for Darwin"
    ccmake .
    make -j 2
    make install
elif [ $cputype == "x86_64" -a $arch == "Linux" ] ; then
    echo "Configuring for 64 bit linux"
else
    echo "Using default configuration"
fi



# finally, remove the builddir

#rm -rf $builddir
