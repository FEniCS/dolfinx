#!/bin/sh

# Generate configure script dolfin-config

# Include variables saved by make
TMPFILE="var.tmp"
COMPILER=`cat $TMPFILE | grep COMPILER | cut -d'"' -f2`
CFLAGS=`cat $TMPFILE | grep CFLAGS | cut -d'"' -f2`
LIBS=`cat $TMPFILE | grep LIBS | cut -d'"' -f2`
PACKAGE=`cat $TMPFILE | grep PACKAGE | cut -d'"' -f2`
VERSION=`cat $TMPFILE | grep VERSION | cut -d'"' -f2`
PREFIX=`cat $TMPFILE | grep PREFIX | cut -d'"' -f2`

# Get top directory
cd ../..
TOPDIR=`pwd`
cd src/config

# Set variables
CFLAGS_SYSTEM="-I$PREFIX/include $CFLAGS"
CFLAGS_DOLFIN="-I$TOPDIR/include $CFLAGS"
LIBS_SYSTEM="-L$PREFIX/lib $LIBS"
LIBS_DOLFIN="-L$TOPDIR/lib $LIBS"
FILE="./dolfin-config"
TEMPLATE="./dolfin-config.template"
DOLFIN_MAIN_INCLUDE="../kernel/main/dolfin.h"

# Find all libraries (.a files)
DOLFIN_LIBS=""
echo "Scanning for DOLFIN libraries..."
for f in `find .. -name '*.a'`; do
	 lib=`echo $f | xargs basename | cut -d'b' -f2-0 | cut -d'.' -f1`
#	 lflag="-l$lib"
	 DOLFIN_LIBS="$DOLFIN_LIBS $f"
#	 LIBS_DOLFIN="$LFLAGS_INSOURCE $lflag"
	 echo "  Found: $lib"
done

# Find all dolfin include files
DOLFIN_INCLUDES=""
echo "Scanning for DOLFIN include files..."
for f in `find .. -name '*.h*' | grep dolfin | grep -v '~'`; do
	 DOLFIN_INCLUDES="$DOLFIN_INCLUDES $f"
	 echo "  Found: $f"
done

# Generate configure script
echo "Generating data for dolfin-config..."
rm -f ./dolfin-config
echo \#!/bin/sh >> $FILE
echo \# config-script for $PACKAGE version $VERSION >> $FILE
echo >> $FILE
echo COMPILER=\"$COMPILER\" >> $FILE
echo CFLAGS=\"$CFLAGS_SYSTEM\" >> $FILE
echo LIBS=\"$LIBS_SYSTEM\" >> $FILE
echo CFLAGS_DOLFIN=\"$CFLAGS_DOLFIN\" >> $FILE
echo LIBS_DOLFIN=\"$LIBS_DOLFIN\" >> $FILE
echo PACKAGE=\"$PACKAGE\" >> $FILE
echo VERSION=\"$VERSION\" >> $FILE
echo >> $FILE
cat $TEMPLATE >> $FILE
chmod 0755 $FILE

# Copy include files and libraries
echo Copying main include file to ../../include
cp $DOLFIN_MAIN_INCLUDE ../../include
echo Copying include files to ../../include/dolfin
cp $DOLFIN_INCLUDES ../../include/dolfin
echo Copying library files to ../../lib
cp $DOLFIN_LIBS ../../lib
