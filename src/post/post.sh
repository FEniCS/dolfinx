#!/bin/sh

# Generate configure script dolfin-config

# Main include file
MAIN_INCLUDE="../kernel/main/dolfin.h"

# Find all libraries (.a files)
DOLFIN_LIBS=""
echo "Scanning for DOLFIN libraries..."
for f in `find .. -name '*.a'`; do
	 lib=`echo $f | xargs basename | cut -d'b' -f2-0 | cut -d'.' -f1`
	 DOLFIN_LIBS="$DOLFIN_LIBS $f"
	 echo "  Found: $lib"
done

# Find all dolfin include files
DOLFIN_INCLUDES=""
echo "Scanning for DOLFIN include files..."
for f in `find .. -name '*.h*' | grep dolfin | grep -v 'dolfin.h' | grep -v '~'`; do
	 DOLFIN_INCLUDES="$DOLFIN_INCLUDES $f"
	 echo "  Found: $f"
done

# Copy include files and libraries
echo Copying main include file to ../../include
cp $MAIN_INCLUDE ../../include
echo Copying include files to ../../include/dolfin
cp $DOLFIN_INCLUDES ../../include/dolfin
echo Copying libraries to ../../lib
cp $DOLFIN_LIBS ../../lib
