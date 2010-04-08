// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef  POINT_3_BBOX_3_INTERSECTION_H
#define  POINT_3_BBOX_3_INTERSECTION_H

#include <CGAL/Bbox_3.h>
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
  bool do_intersect(const typename K::Point_3& pt,
		    const CGAL::Bbox_3& bbox,
		    const K &)
  {
    for(int i = 0; i < 3; ++i)
      if(bbox.max(i) < pt[i] || bbox.min(i) > pt[i])
	return false;
    return true;
  }

  template <class K>
  inline
  bool do_intersect(const CGAL::Bbox_3& bbox,
		    const typename K::Point_3& pt,
		    const K &)
  {
    for(int i = 0; i < 3; ++i)
      if(bbox.max(i) < pt[i] || bbox.min(i) > pt[i])
	return false;
    return true;
  }

template <class K>
inline
Object
intersection(const typename K::Point_3 &pt, 
	     const typename K::Bbox_3 &bbox, 
	     const K&)
{
    if (do_intersect(pt,bbox)) {
        return make_object(pt);
    }
    return Object();
}

template <class K>
inline
Object
intersection( const typename K::Bbox_3 &bbox, 
	      const typename K::Point_3 &pt, 
	      const K&)
{
    if (do_intersect(pt,bbox)) {
        return make_object(pt);
    }
    return Object();
}


} //namespace CGALi

template <class K>
inline
bool do_intersect(const CGAL::Point_3<K>& point,
		  const CGAL::Bbox_3& bbox)
{
  return typename K::Do_intersect_3()(point, bbox);
//  return CGALi::do_intersect(bbox, point, K());
}

template <class K>
inline
bool do_intersect(const CGAL::Bbox_3& bbox,
		  const CGAL::Point_3<K>& point)
{
  return typename K::Do_intersect_3()(point, bbox);
//  return CGALi::do_intersect(bbox, point, K());
}

template <class K>
inline Object
intersection(const Bbox_3 & bbox, const Point_3<K> & pt)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, bbox);
}

template <class K>
inline Object
intersection(const Point_3<K> & pt, const Bbox_3 & bbox)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, bbox);
}

CGAL_END_NAMESPACE

#endif   /* ----- #ifndef POINT_3_BBOX_3_INTERSECTION_H  ----- */
