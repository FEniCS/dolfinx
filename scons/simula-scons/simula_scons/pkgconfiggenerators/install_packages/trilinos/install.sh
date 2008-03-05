#!/bin/bash

prefix=${1:-"/usr/local/trilinos"}

echo "Install Trilinos 8.0.3 in $prefix"

builddir=$(mktemp -d /tmp/trilinosbuild.XXXX)
cd $builddir

curl -O http://trilinos.sandia.gov/download/files/trilinos-8.0.3.tar.gz
tar zxf trilinos-8.0.3.tar.gz
cd trilinos-8.0.3

cputype=$(uname -m)
arch=$(uname -s)

if [ $cputype == "i386" -a $arch == "Darwin" ] ; then
    echo "Configuring for Darwin"
    MACOSX_DEPLOYMENT_TARGET=10.5 \
    CC=gcc-4.0 CCC=g++-4.0 F77=gfortran \
    LDFLAGS='-framework vecLib' \
    ./configure --prefix=$prefix --enable-default-packages \
    --enable-galeri --enable-ml --enable-epetra --enable-amesos --enable-python --enable-shared --enable-pytrilinos
elif [ $cputype == "x86_64" -a $arch == "Linux" ] ; then
    echo "Configuring for 64 bit linux"
    ./configure --prefix=$prefix --enable-default-packages \
    --enable-galeri --enable-ml --enable-epetra --enable-amesos --enable-python --enable-shared \
    --with-cflags=-fPIC --with-cxxflags=-fPIC --with-fflags=-fPIC --with-ldflags=-fPIC --with-ccflags=-fPIC
else
    echo "Using default configuration"
    ./configure --prefix=$prefix --enable-default-packages \
    --enable-galeri --enable-ml --enable-epetra --enable-amesos --enable-python --enable-shared
fi

make
make install

# finally, remove the builddir
rm -rf $builddir
