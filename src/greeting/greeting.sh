#!/bin/sh

# Include variables saved by make
TMPFILE="var.tmp"
VERSION=`cat $TMPFILE | grep VERSION | cut -d'"' -f2`

# Display a message
echo "-------------------------------------------------------------------------------"
echo "                  - DOLFIN successfully compiled -"
echo " "
echo "If you have reached this far, this is a good indication that the compilation"
echo "was successful. The next thing you want to do is probably to try out some of"
echo "the demo programs in the sub-directory src/demo/."
echo " "
echo "To improve future releases of DOLFIN, any feedback or suggestions are very much"
echo "appreciated. Send any comments to dolfin-dev@fenics.org."
#echo " "
#echo "If you like, you can try to type the following command:"
#echo " "
#echo "   uname -a | mail -s 'DOLFIN $VERSION installed' dolfin@math.chalmers.se"
#echo " "
#echo "This will send an email with the output of 'uname -a' to the above address,"
#echo "which will help to give us an indication of the platforms on which people"
#echo "have successfully compiled DOLFIN. Should you be worried about the information"
#echo "you send, type 'uname -a' first to see what it does. The output should be"
#echo "something like"
#echo " "
#echo "   Linux elrond 2.4.19 #4 tis aug 27 09:33:15 CEST 2002 i686 GNU/Linux"
#echo "   SunOS mathtest4 5.7 Generic_106541-16 sun4u sparc SUNW,Ultra-5_1"
echo " "
echo "Enjoy!"
echo "-------------------------------------------------------------------------------"
