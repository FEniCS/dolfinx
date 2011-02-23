// Copyright (C) 2009 Andre Massing
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef  CGAL_POINT_3_POINT_3_INTERSECTION_H
#define	 CGAL_POINT_3_POINT_3_INTERSECTION_H

#include <CGAL/Point_3.h>
#include <CGAL/Object.h>

///@file This file contains some small extension to the CGAL library, for
//instance unifying their do_intersect functions to also deal with Point_3
//and Primitive intersections or some additional intersection collision test.

CGAL_BEGIN_NAMESPACE

#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif

  template <class K >
  inline bool do_intersect(const typename K::Point_3 & pt1,
		    const typename K::Point_3 & pt2,
		    const K & k)
  {
    return  pt1 == pt1;
  }

  template <class K>
  Object
  intersection(const typename K::Point_3 &pt1,
	       const typename K::Point_3 &pt2)
  {
    if (pt1 == pt2)
    {
      return make_object(pt1);
    }
    return Object();
  }


}// namespace CGALi

template <class K>
inline
bool
do_intersect(const Point_3<K> &pt1, const Point_3<K> &pt2)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt1, pt2);
}

template <class K>
inline
Object
intersection(const Point_3<K> &pt1, const Point_3<K> &pt2)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt1, pt2);
}

CGAL_END_NAMESPACE

#endif   // CGAL_POINT_3_POINT_3_INTERSECTION_H
