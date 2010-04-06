// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef CGAL_POINT_3_SEGMENT_3_INTERSECTION_H
#define CGAL_POINT_3_SEGMENT_2_INTERSECTION_H

#include <CGAL/Segment_3.h>
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
	     const typename K::Segment_3 &seg,
	     const K&)
{
    return seg.has_on(pt);
}

template <class K>
inline 
bool
do_intersect(const typename K::Segment_3 &seg,
	     const typename K::Point_3 &pt, 
	     const K&)
{
    return seg.has_on(pt);
}


template <class K>
inline
Object
intersection(const typename K::Point_3 &pt, 
	     const typename K::Segment_3 &seg, 
	     const K&)
{
    if (do_intersect(pt,seg)) {
        return make_object(pt);
    }
    return Object();
}

template <class K>
inline
Object
intersection( const typename K::Segment_3 &seg, 
	      const typename K::Point_3 &pt, 
	      const K&)
{
    if (do_intersect(pt,seg)) {
        return make_object(pt);
    }
    return Object();
}

} // namespace CGALi


template <class K>
inline bool
do_intersect(const Segment_3<K> &seg, const Point_3<K> &pt)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, seg);
}

template <class K>
inline bool
do_intersect(const Point_3<K> &pt, const Segment_3<K> &seg)
{
  typedef typename K::Do_intersect_3 Do_intersect;
    return Do_intersect()(pt, seg);
}


template <class K>
inline Object
intersection(const Segment_3<K> &seg, const Point_3<K> &pt)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, seg);
}

template <class K>
inline Object
intersection(const Point_3<K> &pt, const Segment_3<K> &seg)
{
  typedef typename K::Intersect_3 Intersect;
    return Intersect()(pt, seg);
}

CGAL_END_NAMESPACE

#endif
