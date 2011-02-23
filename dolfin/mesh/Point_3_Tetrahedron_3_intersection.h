// Copyright (C) 2009 Andre Massing
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef CGAL_POINT_3_TETRAHEDRON_3_INTERSECTION_H
#define CGAL_POINT_3_TETRAHEDRON_3_INTERSECTION_H

#include <CGAL/Point_3.h>
#include <CGAL/Object.h>
#include <CGAL/Tetrahedron_3.h>

CGAL_BEGIN_NAMESPACE


#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif

  template <class K>
  inline
  bool
  do_intersect(const typename K::Point_3 &pt,
	       const typename K::Tetrahedron_3 &tet,
	       const K&)
  {
    return !tet.has_on_unbounded_side(pt);
  }

  template <class K>
  inline
  bool
  do_intersect(const typename K::Tetrahedron_3 &tet,
	       const typename K::Point_3 &pt,
	       const K&)
  {
    return !tet.has_on_unbounded_side(pt);
  }


  template <class K>
  inline
  Object
  intersection(const typename K::Point_3 &pt,
	       const typename K::Tetrahedron_3 &tet,
	       const K&)
  {
    if (do_intersect(pt,tet))
    {
      return make_object(pt);
    }
    return Object();
  }

  template <class K>
  inline
  Object
  intersection( const typename K::Tetrahedron_3 &tet,
	        const typename K::Point_3 &pt,
	        const K&)
  {
    if (do_intersect(pt,tet))
    {
      return make_object(pt);
    }
    return Object();
  }

} // namespace CGALi


template <class K>
inline bool
do_intersect(const Tetrahedron_3<K> &tet, const Point_3<K> &pt)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, tet);
}

template <class K>
inline bool
do_intersect(const Point_3<K> &pt, const Tetrahedron_3<K> &tet)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, tet);
}


template <class K>
inline Object
intersection(const Tetrahedron_3<K> &tet, const Point_3<K> &pt)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, tet);
}

template <class K>
inline Object
intersection(const Point_3<K> &pt, const Tetrahedron_3<K> &tet)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, tet);
}

CGAL_END_NAMESPACE

#endif
