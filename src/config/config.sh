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
