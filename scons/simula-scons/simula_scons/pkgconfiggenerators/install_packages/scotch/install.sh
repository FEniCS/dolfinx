#!/bin/bash

prefix=${1:-"/usr/local/scotch"}

echo "Install scotch-4.0 in $prefix"

builddir=$(mktemp -dt scotchbuild)
cd $builddir
# Make gmake a symlink to make in this directory, and add this dir first in
# PATH
ln -sf $(type -p make) gmake
PATH=`pwd`:$PATH

wget http://gforge.inria.fr/frs/download.php/464/scotch_4.0.0.tar.gz
tar zxf scotch_4.0.0.tar.gz
cd scotch_4.0/src

cputype=$(uname -m)
arch=$(uname -s)

if [ $cputype == "i686" -a $arch == "Linux" ] ; then
   ln -sf Make.inc/Makefile.inc.i686_pc_linux2 Makefile.inc
else
   ln -sf Make.inc/Makefile.inc.${cputype}_${arch} Makefile.inc
fi

make
cd ../bin

[ -d $prefix ] || mkdir -p $prefix
[ -d $prefix/lib ] || mkdir $prefix/lib
[ -d $prefix/include ] || mkdir $prefix/include

cp lib*.a $prefix/lib/.
cp *.h $prefix/include/.

ls -l $prefix/lib
ls -l $prefix/include

# finally, remove the builddir
rm -rf $builddir
