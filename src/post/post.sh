#!/bin/sh

# Generate configure script dolfin-config

cd ../../
TOPLEVEL=`pwd`

# Main include file
MAIN_INCLUDE="src/kernel/main/dolfin.h"

# Find all dolfin include files
DOLFIN_INCLUDES=""
echo "Scanning for DOLFIN include files..."
for f in `find src -name '*.h*' | grep dolfin | grep -v 'dolfin.h' | grep -v '~'`; do
	 DOLFIN_INCLUDES="$DOLFIN_INCLUDES $f"
#	 echo "  Found: $f"
done

# Find all libraries (.a files)
DOLFIN_LIBS=""
echo "Scanning for DOLFIN libraries..."
for f in `find src -name '*.a'`; do
	 lib=`echo $f | xargs basename | cut -d'b' -f2-0 | cut -d'.' -f1`
	 DOLFIN_LIBS="$DOLFIN_LIBS $f"
	 echo "  Found: $lib"
done

# Create symbolic link for main include file
ln -s -f $TOPLEVEL/$MAIN_INCLUDE include

# Create symbolic links for include files
for f in $DOLFIN_INCLUDES; do
    ln -s -f $TOPLEVEL/$f include/dolfin
done

# Create symbolic links for libraries
for f in $DOLFIN_LIBS; do
    ln -s -f $TOPLEVEL/$f lib
done
