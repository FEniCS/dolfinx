#!/bin/sh

# Generate configure script dolfin-config

# Include variables saved by make
TMPFILE="var.tmp"
COMPILER=`cat $TMPFILE | grep COMPILER | cut -d'"' -f2`
CFLAGS=`cat $TMPFILE | grep CFLAGS | cut -d'"' -f2`
LFLAGS=`cat $TMPFILE | grep LFLAGS | cut -d'"' -f2`
PACKAGE=`cat $TMPFILE | grep PACKAGE | cut -d'"' -f2`
VERSION=`cat $TMPFILE | grep VERSION | cut -d'"' -f2`

# Get top directory
cd ../..
TOPDIR=`pwd`
cd src/config

# Set variables
CFLAGS_INSOURCE="-I$TOPDIR/include $CFLAGS"
LFLAGS_INSOURCE="-L$TOPDIR/lib $LFLAGS"
CFLAGS="not_configured"
LFLAGS="not_configured"
FILE="./dolfin-config"
TEMPLATE="./dolfin-config.template"
DOLFIN_INCLUDES="../main/dolfin.h"

# Find all libraries (.a files)
DOLFIN_LIBS=""
echo "Scanning for DOLFIN libraries..."
for f in `find .. -name '*.a'`; do
	 lib=`echo $f | xargs basename | cut -d'b' -f2 | cut -d'.' -f1`
#	 lflag="-l$lib"
	 DOLFIN_LIBS="$DOLFIN_LIBS $f"
#	 LFLAGS_INSOURCE="$LFLAGS_INSOURCE $lflag"
	 echo "  Found: $lib"
done

# Generate configure script
echo "Generating data for dolfin-config..."
rm -f ./dolfin-config
echo \#!/bin/sh >> $FILE
echo \# config-script for $PACKAGE version $VERSION >> $FILE
echo >> $FILE
echo COMPILER=\"$COMPILER\" >> $FILE
echo CFLAGS=\"$CFLAGS\" >> $FILE
echo LFLAGS=\"$LFLAGS\" >> $FILE
echo CFLAGS_INSOURCE=\"$CFLAGS_INSOURCE\" >> $FILE
echo LFLAGS_INSOURCE=\"$LFLAGS_INSOURCE\" >> $FILE
echo PACKAGE=\"$PACKAGE\" >> $FILE
echo VERSION=\"$VERSION\" >> $FILE
echo >> $FILE
cat $TEMPLATE >> $FILE
chmod 0755 $FILE

# Copy include files and libraries
echo Copying include files to ../../include
cp $DOLFIN_INCLUDES ../../include
echo Copying library files to ../../lib
cp $DOLFIN_LIBS ../../lib
