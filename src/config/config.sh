#!/bin/sh

# Generate configure script dolfin-config

# Include variables saved by make
TMPFILE="var.tmp"
COMPILER=`cat $TMPFILE | grep COMPILER | cut -d'"' -f2`
LINKER=`cat $TMPFILE | grep LINKER | cut -d'"' -f2`
CFLAGS=`cat $TMPFILE | grep CFLAGS | cut -d'"' -f2`
INCLUDES=`cat $TMPFILE | grep INCLUDES | cut -d'"' -f2`
LIBS=`cat $TMPFILE | grep LIBS | cut -d'"' -f2`
PYTHON_CPPFLAGS=`cat $TMPFILE | grep PYTHON_CPPFLAGS | cut -d'"' -f2`
PACKAGE=`cat $TMPFILE | grep PACKAGE | cut -d'"' -f2`
VERSION=`cat $TMPFILE | grep VERSION | cut -d'"' -f2`
PREFIX=`cat $TMPFILE | grep PREFIX | cut -d'"' -f2`

# Get top directory
cd ../..
TOPDIR=`pwd`
cd src/config

# Set variables
CFLAGS_SYSTEM="-I$PREFIX/include $CFLAGS"
LIBS_SYSTEM="-L$PREFIX/lib $LIBS"
INCLUDES_SYSTEM="-I$PREFIX/include $INCLUDES"
FILE="./dolfin-config"
TEMPLATE="./dolfin-config.template"

# Generate configure script
echo "Generating data for dolfin-config..."
rm -f ./dolfin-config
echo \#!/bin/sh >> $FILE
echo \# config-script for $PACKAGE version $VERSION >> $FILE
echo >> $FILE
echo COMPILER=\"$COMPILER\" >> $FILE
echo LINKER=\"$LINKER\" >> $FILE
echo CFLAGS=\"$CFLAGS_SYSTEM\" >> $FILE
echo INCLUDES=\"$INCLUDES_SYSTEM\" >> $FILE
echo LIBS=\"$LIBS_SYSTEM\" >> $FILE
echo SWIGCFLAGS=\"$PYTHON_CPPFLAGS\" >> $FILE
echo PACKAGE=\"$PACKAGE\" >> $FILE
echo VERSION=\"$VERSION\" >> $FILE
echo >> $FILE
cat $TEMPLATE >> $FILE
chmod 0755 $FILE
