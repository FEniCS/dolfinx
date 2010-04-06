// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef  segment_3_tetrahedron_3_intersection_INC
#define  segment_3_tetrahedron_3_intersection_INC

#include <CGAL/Segment_3.h>
#include <CGAL/Tetrahedron_3.h>
#include <CGAL/Object.h>

#include <dolfin/log/log.h>

using dolfin::error;

CGAL_BEGIN_NAMESPACE

#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif

  template <class K>
  inline 
  bool
  do_intersect(const typename K::Tetrahedron_3 &tet, 
               const typename K::Segment_3 &seg,
               const K&)
  {
    //throw exception!
    dolfin_not_implemented();

    return false;
  }

  template <class K>
  inline 
  bool
  do_intersect(const typename K::Segment_3 &seg,
               const typename K::Tetrahedron_3 &tet, 
               const K&)
  {
    //throw exception!
    dolfin_not_implemented();

    return false;
  }


  template <class K>
  inline
  Object
  intersection(const typename K::Tetrahedron_3 &tet, 
               const typename K::Segment_3 &seg, 
               const K&)
  {
    //throw exception!
    dolfin_not_implemented();

    if (do_intersect(tet,seg)) {
      return Object();
    }
    return Object();
  }

  template <class K>
  inline
  Object
  intersection( const typename K::Segment_3 &seg, 
                const typename K::Tetrahedron_3 &tet, 
                const K&)
  {
    //throw exception!
    dolfin_not_implemented();

    if (do_intersect(tet,seg)) {
      return Object();
    }
    return Object();
  }

} // namespace CGALi


template <class K>
  inline bool
do_intersect(const Segment_3<K> &seg, const Tetrahedron_3<K> &tet)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(tet, seg);
}

template <class K>
  inline bool
do_intersect(const Tetrahedron_3<K> &tet, const Segment_3<K> &seg)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(tet, seg);
}


template <class K>
  inline Object
intersection(const Segment_3<K> &seg, const Tetrahedron_3<K> &tet)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(tet, seg);
}

template <class K>
  inline Object
intersection(const Tetrahedron_3<K> &tet, const Segment_3<K> &seg)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(tet, seg);
}

CGAL_END_NAMESPACE


#endif   /* ----- #ifndef segment_3_tetrahedron_3_intersection_INC  ----- */
