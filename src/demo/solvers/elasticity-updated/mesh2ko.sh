#!/bin/sh

for a in $*
  do
  
  #echo "${a%.m}; mesh2ko(points, edges, cells, '${a%.m}.xml')" | octave
  ko-mesh2phys $a ko-${a%.gz}

  done
