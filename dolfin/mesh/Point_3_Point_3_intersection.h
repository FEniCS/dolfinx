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
