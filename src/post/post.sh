#!/bin/sh

# Generate configure script dolfin-config

# Find all libraries (.a files)
DOLFIN_LIBS=""
echo "Scanning for DOLFIN libraries..."
for f in `find .. -name '*.a'`; do
	 lib=`echo $f | xargs basename | cut -d'b' -f2-0 | cut -d'.' -f1`
	 DOLFIN_LIBS="$DOLFIN_LIBS $f"
	 echo "  Found: $lib"
done

# Find all dolfin include files
DOLFIN_INCLUDES="../kernel/main/dolfin.h"
echo "Scanning for DOLFIN include files..."
for f in `find .. -name '*.h*' | grep dolfin | grep -v 'dolfin.h' | grep -v '~'`; do
	 DOLFIN_INCLUDES="$DOLFIN_INCLUDES $f"
	 echo "  Found: $f"
done

# Copy include files and libraries
echo Copying include files to ../../include/dolfin
cp $DOLFIN_INCLUDES ../../include/dolfin
echo Copying library files to ../../lib
cp $DOLFIN_LIBS ../../lib
