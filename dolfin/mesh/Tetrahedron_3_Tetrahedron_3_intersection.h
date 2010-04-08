// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef  TETRAHEDRON_3_TETRAHEDRON_3_INTERSECTION_INC
#define  TETRAHEDRON_3_TETRAHEDRON_3_INTERSECTION_INC

#include <CGAL/Tetrahedron_3.h>
//#include <CGAL/Triangle_3_Tetrahedron_3_do_intersect.h>

CGAL_BEGIN_NAMESPACE

#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif

  //This code is not optimized!!
  template <class K >
  bool
  do_intersect(const typename K::Tetrahedron_3 & tet1, 
               const typename K::Tetrahedron_3 & tet2,
               const K & k)
  {
    typedef typename K::Triangle_3 Triangle;

    //Check first whether on point of one primitive intersect the other
    //Check for all points, might be more efficient...?
    if (!tet1.has_on_unbounded_side(tet2[0])) return true;
    if (!tet1.has_on_unbounded_side(tet2[1])) return true;
    if (!tet1.has_on_unbounded_side(tet2[2])) return true;
    if (!tet1.has_on_unbounded_side(tet2[3])) return true;

    if (!tet2.has_on_unbounded_side(tet1[0])) return true;
    if (!tet2.has_on_unbounded_side(tet1[1])) return true;
    if (!tet2.has_on_unbounded_side(tet1[2])) return true;
    if (!tet2.has_on_unbounded_side(tet1[3])) return true;

//    if (!k.has_on_unbounded_side_3_object()(tet1,tet2[0])) return true;
//    if (!k.has_on_unbounded_side_3_object()(tet2,tet1[0])) return true;

    //Otherwise one tetrahedron face must intersect the bbox in order to intersect.
    if (do_intersect(tet1, Triangle(tet2[0], tet2[1], tet2[2]))) return true;
    if (do_intersect(tet1, Triangle(tet2[0], tet2[1], tet2[3]))) return true;
    if (do_intersect(tet1, Triangle(tet2[0], tet2[2], tet2[3]))) return true;
    if (do_intersect(tet1, Triangle(tet2[1], tet2[2], tet2[3]))) return true;

    return false;
  }

}// namespace CGALi

template <class K>
inline bool
do_intersect(const Tetrahedron_3<K> &tet1, const Tetrahedron_3<K> &tet2)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(tet1, tet2);
}

CGAL_END_NAMESPACE


#endif   /* ----- #ifndef TETRAHEDRON_3_TETRAHEDRON_3_INTERSECTION_INC  ----- */
