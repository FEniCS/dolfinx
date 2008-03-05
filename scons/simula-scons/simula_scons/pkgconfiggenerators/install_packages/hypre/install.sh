#!/bin/bash

prefix=${1:-"/usr/local/hypre"}

echo "Install Hypre 2.0.0 in $prefix"

builddir=$(mktemp -dt hyprebuild)
#copy files needed by Mac OS X
cp hypre-* $builddir
cd $builddir

curl -O -k https://computation.llnl.gov/casc/hypre/download/hypre-2.0.0.tar.gz
tar xzf hypre-2.0.0.tar.gz 

cd hypre-2.0.0/src

cputype=$(uname -m)
arch=$(uname -s)

if [ $cputype == "i386" -a $arch == "Darwin" ] ; then
    #copy files that are needed by mac os x
    rm -f configure
    cp ../../hypre-configure configure
    chmod u+x configure
    cp ../../hypre-struct_overlap_innerprod.c struct_mv/struct_overlap_innerprod.c 
    echo "Configuring for Darwin"
    MACOSX_DEPLOYMENT_TARGET=10.5
    LDFLAGS='-framework vecLib' 
    ./configure --without-MPI --enable-shared --prefix=$prefix --without-fei --without-superlu --without-mli --with-blas --with-lapack
elif [ $cputype == "x86_64" -a $arch == "Linux" ] ; then
    echo "Configuring for 64 bit linux"
    CFLAGS=-fPIC
    CXXFLAGS=-fPIC
    FFLAGS=-fPIC
    LDFLAGS=-fPIC
    ./configure --without-MPI --enable-shared --prefix=$prefix --without-fei --without-superlu --without-mli --with-blas --with-lapack
else
    echo "Using default configuration"
    ./configure --without-MPI --enable-shared --prefix=$prefix --without-fei --without-superlu --without-mli --with-blas --with-lapack
fi

make
make install


# finally, remove the builddir

rm -rf $builddir
