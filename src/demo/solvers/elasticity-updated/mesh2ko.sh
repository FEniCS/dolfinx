#!/bin/sh

for a in $*
  do
  
  echo "${a%.m}; mesh2ko(points, edges, cells, '${a%.m}.xml')" | octave

  done
