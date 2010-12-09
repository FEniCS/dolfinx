#!/bin/sh
#
# This script plots a bunch of rotating elements.
#
# Anders Logg, 2010-12-08

echo "Plotting elements..."

dolfin-plot Argyris                triangle     5 rotate=1 &
dolfin-plot Arnold-Winther         triangle       rotate=1 &
dolfin-plot Brezzi-Douglas-Marini  tetrahedron  3 rotate=1 &
dolfin-plot Crouzeix-Raviart       triangle     1 rotate=1 &
dolfin-plot Hermite                triangle       rotate=1 &
dolfin-plot Hermite                tetrahedron    rotate=1 &
dolfin-plot Lagrange               tetrahedron  5 rotate=1 &
dolfin-plot Mardal-Tai-Winther     triangle       rotate=1 &
dolfin-plot Morley                 triangle       rotate=1 &
dolfin-plot N1curl                 tetrahedron  5 rotate=1 &
dolfin-plot Raviart-Thomas         tetrahedron  1 rotate=1 &
