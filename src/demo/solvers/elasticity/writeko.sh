#!/bin/sh

for a in $*
  do
    echo "${a%.m}; writeko(points, edges, cells, u);" | octave
  done
