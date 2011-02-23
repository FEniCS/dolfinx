// Copyright (C) 2009 Andre Massing
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2009-11-10

#ifndef CGAL_POINT_3_TRIANGLE_3_INTERSECTION_H
#define CGAL_POINT_3_TRIANGLE_3_INTERSECTION_H

#include <CGAL/Point_3.h>
#include <CGAL/Triangle_3.h>
#include <CGAL/Object.h>

//already defined in original
//CGAL, but not with p.has_on(pt) function??

#ifdef SUBSTITUTE_ORIGINAL_CGAL_FUNCTION

CGAL_BEGIN_NAMESPACE

namespace CGALi
{

  template <class K>
  inline
  bool
  do_intersect(const typename K::Point_3 &pt,
	       const typename K::Triangle_3 &tri,
	       const K&)
  {
    return tri.has_on(pt);
  }

  template <class K>
  inline
  bool
  do_intersect(const typename K::Triangle_3 &tri,
	       const typename K::Point_3 &pt,
	       const K&)
  {
    return tri.has_on(pt);
  }


  template <class K>
  inline
  Object
  intersection(const typename K::Point_3 &pt,
	       const typename K::Triangle_3 &tri,
	       const K&)
  {
    if (do_intersect(pt,tri))
    {
      return make_object(pt);
    }
    return Object();
  }

  template <class K>
  inline
  Object
  intersection( const typename K::Triangle_3 &tri,
	        const typename K::Point_3 &pt,
	        const K&)
  {
    if (do_intersect(pt,tri))
    {
      return make_object(pt);
    }
    return Object();
  }

} // namespace CGALi


template <class K>
inline bool
do_intersect(const Triangle_3<K> &tri, const Point_3<K> &pt)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, tri);
}

template <class K>
inline bool
do_intersect(const Point_3<K> &pt, const Triangle_3<K> &tri)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, tri);
}

template <class K>
inline Object
intersection(const Triangle_3<K> &tri, const Point_3<K> &pt)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, tri);
}

template <class K>
inline Object
intersection(const Point_3<K> &pt, const Triangle_3<K> &tri)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, tri);
}

CGAL_END_NAMESPACE

#endif
#endif
