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

#ifndef CGAL_POINT_3_LINE_3_INTERSECTION_H
#define CGAL_POINT_3_LINE_3_INTERSECTION_H

#include <CGAL/Line_3.h>
#include <CGAL/Point_3.h>
#include <CGAL/Object.h>

CGAL_BEGIN_NAMESPACE

#if CGAL_VERSION_NR < 1030601000
namespace CGALi {
#else
namespace internal {
#endif

  template <class K>
  inline bool
  do_intersect(const typename K::Point_3 &pt,
	       const typename K::Line_3 &line,
	       const K&)
  {
    return line.has_on(pt);
  }

  template <class K>
  inline bool
  do_intersect(const typename K::Line_3 &line,
	       const typename K::Point_3 &pt,
	       const K&)
  {
    return line.has_on(pt);
  }

  template <class K>
  Object
  intersection(const typename K::Point_3 &pt,
	       const typename K::Line_3 &line,
	       const K& k)
  {
    if (do_intersect(pt, line, k))
    {
      return make_object(pt);
    }
    return Object();
  }

  template <class K>
  Object
  intersection(const typename K::Line_3 &line,
	       const typename K::Point_3 &pt,
	       const K& k)
  {
    if (do_intersect(pt, line, k))
    {
      return make_object(pt);
    }
    return Object();
  }

} // namespace CGALi

template <class K>
inline
bool
do_intersect(const Line_3<K> &line,
	     const Point_3<K> &pt)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, line);
}

template <class K>
inline
bool
do_intersect(const Point_3<K> &pt,
	     const Line_3<K> &line)
{
    typedef typename K::Do_intersect_3 Do_intersect;
    return Do_intersect()(pt, line);
}

template <class K>
inline
Object
intersection(const Line_3<K> &line,
	     const Point_3<K> &pt)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, line);
}

template <class K>
inline
Object
intersection(const Point_3<K> &pt,
	     const Line_3<K> &line)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, line);
}

CGAL_END_NAMESPACE

#endif
