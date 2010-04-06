// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef CGAL_POINT_3_RAY_3_INTERSECTION_H
#define CGAL_POINT_3_RAY_3_INTERSECTION_H

#include <CGAL/Ray_3.h>
#include <CGAL/Point_3.h>
#include <CGAL/Object.h>

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
	     const typename K::Ray_3 &ray,
	     const K&)
{
  return ray.has_on(pt);
}


template <class K>
inline 
bool
do_intersect(const typename K::Ray_3 &ray,
	     const typename K::Point_3 &pt, 
	     const K&)
{
  return ray.has_on(pt);
}


template <class K>
Object
intersection(const typename K::Point_3 &pt, 
	     const typename K::Ray_3 &ray,
	     const K& k)
{
  if (do_intersect(pt,ray, k)) {
    return make_object(pt);
  }
  return Object();
}

template <class K>
Object
intersection(const typename K::Ray_3 &ray,
	     const typename K::Point_3 &pt, 
	     const K& k)
{
  if (do_intersect(pt,ray, k)) {
    return make_object(pt);
  }
  return Object();
}

} // namespace CGALi


template <class K>
inline
bool
do_intersect(const Ray_3<K> &ray, const Point_3<K> &pt)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, ray);
}

template <class K>
inline
bool
do_intersect(const Point_3<K> &pt, const Ray_3<K> &ray)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, ray);
}


template <class K>
inline Object
intersection(const Ray_3<K> &ray, const Point_3<K> &pt)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, ray);
}

template <class K>
inline Object
intersection(const Point_3<K> &pt, const Ray_3<K> &ray)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, ray);
}

CGAL_END_NAMESPACE

#endif
