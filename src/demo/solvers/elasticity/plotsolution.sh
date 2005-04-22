#!/bin/sh

#./writeko.sh elasticity.m

if [ "´which ko-render´" != "" ] && [ "´which ko-mesh2phys´" != "" ] 
then
    for a in mesh0*.xml.gz
    do
      ko-mesh2phys $a ${a%.gz}
    done

    ko-render template.pov mesh0*.xml
else
    echo "Cannot find rendering tools \"ko-render\" or \"ko-mesh2phys\"."
fi
