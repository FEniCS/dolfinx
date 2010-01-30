// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-16
// Last changed: 2010-01-27

#ifndef  primitives_traits_INC
#define  primitives_traits_INC

#ifdef HAS_CGAL

#include "Vertex.h"
#include "MeshEntityIterator.h"
#include "MeshEntity.h"
#include "PointCell.h"
#include "IntervalCell.h"
#include "TriangleCell.h"
#include "TetrahedronCell.h"

namespace dolfin {

struct PointPrimitive {};

///Forward declaration for a general Traits class. This traits class is
///supposed to provide a datum function, which returns a geometric primitive
///object, which type corresponds to the primitive type (Point, PointCell,
///Tetrahedron(Cell) etc.) and the passed geometric CGAL kernel.
template <typename Primitive, typename Kernel> struct Primitive_Traits;

template <typename Kernel> struct Primitive_Traits<PointPrimitive,Kernel> {
  typedef Kernel K;
  typedef PointPrimitive Primitive;
  typedef typename K::Point_3 Datum;
  static const int dim = 0;
  static Datum datum(const Point & point) {
    return Datum(point);
  }
};

template <typename Kernel> struct Primitive_Traits<PointCell,Kernel> {
  typedef Kernel K;
  typedef PointCell Primitive;
  typedef typename K::Point_3 Datum;
  static const int dim = 0;
  static Datum datum(const MeshEntity & cell) {
    VertexIterator v(cell);
    return Datum(v->point());
  }
};

template <typename Kernel> struct Primitive_Traits<IntervalCell,Kernel> {
  typedef Kernel K;
  typedef IntervalCell Primitive;
  typedef typename K::Point_3 Point_3;
  typedef typename K::Segment_3 Datum;
  static const int dim = 1;
  static Datum datum(const MeshEntity & cell) {
    VertexIterator v(cell);
    Point_3 p1(v->point());
    ++v;
    Point_3 p2(v->point());
    return Datum(p1,p2);
  }
};

template <typename Kernel> struct Primitive_Traits<TriangleCell,Kernel> {
  typedef Kernel K;
  typedef TriangleCell Primitive;
  typedef typename K::Point_3 Point_3;
  typedef typename K::Triangle_3 Datum;
  static const int dim = 2;
  static Datum datum(const MeshEntity & cell) {
    VertexIterator v(cell);
    Point_3 p1(v->point());
    ++v;
    Point_3 p2(v->point());
    ++v;
    Point_3 p3(v->point());
    return Datum(p1,p2,p3);
  }
};

template <typename Kernel> struct Primitive_Traits<TetrahedronCell,Kernel> {
  typedef Kernel K;
  typedef TetrahedronCell Primitive;
  typedef typename K::Point_3 Point_3;
  typedef typename K::Tetrahedron_3 Datum;
  static const int dim = 3;
  static Datum datum(const MeshEntity & cell) {
    VertexIterator v(cell);
    Point_3 p1(v->point());
    ++v;
    Point_3 p2(v->point());
    ++v;
    Point_3 p3(v->point());
    ++v;
    Point_3 p4(v->point());
    return Datum(p1,p2,p3,p4);
  }
};

} //end namespace dolfin

#endif   
#endif   /* ----- #ifndef primitives_traits_INC  ----- */
