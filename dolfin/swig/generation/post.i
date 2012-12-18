/* -*- C -*- */
// ===========================================================================
// SWIG directives for the DOLFIN generation kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
// ===========================================================================

%extend dolfin::CSGGeometry {
  %pythoncode %{
     def __add__(self, other) :
         return CSGUnion(self, other)

     def __mul__(self, other) :
         return CSGIntersection(self, other)

     def __sub__(self, other) :
         return CSGDifference(self, other)

  %}
 }
