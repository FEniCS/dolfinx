// Copyright (C) 2009 Andre Massing
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-09-11
// Last changed: 2010-04-06

#ifndef  segment_3_segment_3_intersection_INC
#define  segment_3_segment_3_intersection_INC

#include <CGAL/version.h>

///@file This file contains some small extension to the CGAL library, for
//instance unifying their do_intersect functions to also deal with Segment_3
//and Primitive intersections or some additional intersection collision test.
// It is not required for CGAL_VERSION >= 3.8

#if CGAL_VERSION_NR < 1030801000

#include <CGAL/Segment_3.h>
#include <CGAL/Object.h>

#include <dolfin/log/log.h>

using dolfin::error;

CGAL_BEGIN_NAMESPACE

#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif


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


#endif

#endif   /* ----- #ifndef segment_3_segment_3_intersection_INC  ----- */
