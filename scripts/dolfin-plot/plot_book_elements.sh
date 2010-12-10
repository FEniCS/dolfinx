#!/bin/sh
#
# This script plots a bunch of elements. Images produced by this
# script are used in the FEniCS book chapter "Common and unusual
# finite elements" by Kirby/Logg/Rognes/Terrel.
#
# Anders Logg, 2010-12-08

ROTATE=0

dolfin-plot Argyris                triangle     5 rotate=$ROTATE

dolfin-plot Arnold-Winther         triangle       rotate=$ROTATE

dolfin-plot Brezzi-Douglas-Marini  triangle     1 rotate=$ROTATE
dolfin-plot Brezzi-Douglas-Marini  triangle     2 rotate=$ROTATE
dolfin-plot Brezzi-Douglas-Marini  triangle     3 rotate=$ROTATE
dolfin-plot Brezzi-Douglas-Marini  tetrahedron  1 rotate=$ROTATE
dolfin-plot Brezzi-Douglas-Marini  tetrahedron  2 rotate=$ROTATE
dolfin-plot Brezzi-Douglas-Marini  tetrahedron  3 rotate=$ROTATE

dolfin-plot Crouzeix-Raviart       triangle     1 rotate=$ROTATE
dolfin-plot Crouzeix-Raviart       tetrahedron  1 rotate=$ROTATE

dolfin-plot DG                     triangle     0 rotate=$ROTATE
dolfin-plot DG                     triangle     1 rotate=$ROTATE
dolfin-plot DG                     triangle     2 rotate=$ROTATE
dolfin-plot DG                     triangle     3 rotate=$ROTATE
dolfin-plot DG                     tetrahedron  0 rotate=$ROTATE
dolfin-plot DG                     tetrahedron  1 rotate=$ROTATE
dolfin-plot DG                     tetrahedron  2 rotate=$ROTATE
dolfin-plot DG                     tetrahedron  3 rotate=$ROTATE

dolfin-plot Hermite                triangle       rotate=$ROTATE
dolfin-plot Hermite                tetrahedron    rotate=$ROTATE

dolfin-plot Lagrange               triangle     1 rotate=$ROTATE
dolfin-plot Lagrange               triangle     2 rotate=$ROTATE
dolfin-plot Lagrange               triangle     3 rotate=$ROTATE
dolfin-plot Lagrange               triangle     4 rotate=$ROTATE
dolfin-plot Lagrange               triangle     5 rotate=$ROTATE
dolfin-plot Lagrange               triangle     6 rotate=$ROTATE
dolfin-plot Lagrange               tetrahedron  1 rotate=$ROTATE
dolfin-plot Lagrange               tetrahedron  2 rotate=$ROTATE
dolfin-plot Lagrange               tetrahedron  3 rotate=$ROTATE
dolfin-plot Lagrange               tetrahedron  4 rotate=$ROTATE
dolfin-plot Lagrange               tetrahedron  5 rotate=$ROTATE
dolfin-plot Lagrange               tetrahedron  6 rotate=$ROTATE

dolfin-plot Mardal-Tai-Winther     triangle       rotate=$ROTATE

dolfin-plot Morley                 triangle       rotate=$ROTATE

dolfin-plot N1curl                 triangle     1 rotate=$ROTATE
dolfin-plot N1curl                 triangle     2 rotate=$ROTATE
dolfin-plot N1curl                 triangle     3 rotate=$ROTATE
dolfin-plot N1curl                 tetrahedron  1 rotate=$ROTATE
dolfin-plot N1curl                 tetrahedron  2 rotate=$ROTATE
dolfin-plot N1curl                 tetrahedron  3 rotate=$ROTATE

dolfin-plot N2curl                 triangle     1 rotate=$ROTATE
dolfin-plot N2curl                 triangle     2 rotate=$ROTATE
dolfin-plot N2curl                 triangle     3 rotate=$ROTATE
dolfin-plot N2curl                 tetrahedron  1 rotate=$ROTATE

dolfin-plot Raviart-Thomas         triangle     1 rotate=$ROTATE
dolfin-plot Raviart-Thomas         triangle     2 rotate=$ROTATE
dolfin-plot Raviart-Thomas         triangle     3 rotate=$ROTATE
dolfin-plot Raviart-Thomas         tetrahedron  1 rotate=$ROTATE
dolfin-plot Raviart-Thomas         tetrahedron  2 rotate=$ROTATE
dolfin-plot Raviart-Thomas         tetrahedron  3 rotate=$ROTATE
