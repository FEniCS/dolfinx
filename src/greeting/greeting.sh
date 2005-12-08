#!/bin/sh

# Include variables saved by make
TMPFILE="var.tmp"
VERSION=`cat $TMPFILE | grep VERSION | cut -d'"' -f2`

# Display a message
echo "-------------------------------------------------------------------------------"
echo "                  - DOLFIN successfully compiled -"
echo " "
echo "The next thing you want to do is probably to try out some of the demo programs"
echo "in the sub-directory src/demo. To compile the demos, you first need to install"
echo "DOLFIN on your system. Type"
echo ""
echo "    make install"
echo ""
echo "to install DOLFIN in the directory specified earlier when running configure."
echo "If you did not specify any directory, then DOLFIN will be installed in the"
echo "system's default installation directory (which is most likely /usr/local/)"
echo ""
echo "After you have installed DOLFIN, the demo programs can be compiled by running"
echo ""
echo "    make demo"
echo " "
echo "If you are new to DOLFIN, a good place to start is in src/demo/poisson."
echo "                                                      ----------------"
echo " "
echo "To improve future releases of DOLFIN, any feedback or suggestions are very much"
echo "appreciated. Send any comments to dolfin-dev@fenics.org."
echo " "
echo "Enjoy!"
echo "-------------------------------------------------------------------------------"
