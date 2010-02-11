// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-02-10

#ifndef  segment_3_segment_3_intersection_INC
#define  segment_3_segment_3_intersection_INC


#include <CGAL/Segment_3.h>
#include <CGAL/Object.h>

#include <dolfin/log/log.h>

using dolfin::error;

///@file This file contains some small extension to the CGAL library, for
//instance unifying their do_intersect functions to also deal with Segment_3
//and Primitive intersections or some additional intersection collision test.

CGAL_BEGIN_NAMESPACE

namespace CGALi {


  template <class K >
  inline bool do_intersect(const typename K::Segment_3 & s1, 
                           const typename K::Segment_3 & s2,
                           const K & k)
  {
    //throw exception
    dolfin_not_implemented();
    return  false;
  }

  template <class K>
  inline
  Object
  intersection(const typename K::Segment_3 &s1, 
               const typename K::Segment_3 &s2, 
               const K&)
  {
    //throw exception
    dolfin_not_implemented();

    if (do_intersect(s1,s2)) {
      return Object();
    }
    return Object();
  }


}// namespace CGALi

template <class K>
inline bool
do_intersect(const Segment_3<K> &s1, const Segment_3<K> &s2)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(s1, s2);
}

template <class K>
inline
  Object
intersection(const Segment_3<K> &s1, const Segment_3<K> &s2)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(s1, s2);
}

CGAL_END_NAMESPACE


#endif   /* ----- #ifndef segment_3_segment_3_intersection_INC  ----- */
