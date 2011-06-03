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
// Last changed: 2009-11-10

#ifndef CGAL_POINT_3_ISO_CUBOID_3_INTERSECTION_H
#define CGAL_POINT_3_ISO_CUBOID_3_INTERSECTION_H

#include <CGAL/Iso_cuboid_3.h>
#include <CGAL/Point_3.h>
#include <CGAL/Object.h>

CGAL_BEGIN_NAMESPACE

namespace CGALi
{

template <class K>
inline
bool
do_intersect(const typename K::Point_3 &pt,
	     const typename K::Iso_cuboid_3 &iso,
	     const K&)
{
  return !iso.has_on_unbounded_side(pt);
}

template <class K>
inline
bool
do_intersect(const typename K::Iso_cuboid_3 &iso,
	     const typename K::Point_3 &pt,
	     const K&)
{
  return !iso.has_on_unbounded_side(pt);
}

template <class K>
Object
intersection(const typename K::Point_3 &pt,
	     const typename K::Iso_cuboid_3 &iso,
	     const K& k)
{
  if (CGALi::do_intersect(pt, iso, k))
  {
    return make_object(pt);
  }
  return Object();
}


template <class K>
Object
intersection(const typename K::Iso_cuboid_3 &iso,
	     const typename K::Point_3 &pt,
	     const K& k)
{
  if (CGALi::do_intersect(pt, iso, k))
  {
    return make_object(pt);
  }
  return Object();
}

} // namespace CGALi


template <class K>
inline
bool
do_intersect(const Iso_cuboid_3<K> &iso,
	     const Point_3<K> &pt)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, iso);
}

template <class K>
inline
bool
do_intersect(const Point_3<K> &pt,
	     const Iso_cuboid_3<K> &iso)
{
  typedef typename K::Do_intersect_3 Do_intersect;
  return Do_intersect()(pt, iso);
}

template <class K>
inline
Object
intersection(const Iso_cuboid_3<K> &iso,
	     const Point_3<K> &pt)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, iso);
}

template <class K>
inline
Object
intersection(const Point_3<K> &pt,
	     const Iso_cuboid_3<K> &iso)
{
  typedef typename K::Intersect_3 Intersect;
  return Intersect()(pt, iso);
}

CGAL_END_NAMESPACE

#endif
