// Copyright (C) 2013 Andre Massing
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
// First added:  2013-04-22
// Last changed: 2013-04-23

#ifndef NEAREST_POINT_TETRAHEDRON_3_H_
#define NEAREST_POINT_TETRAHEDRON_3_H_

#include <CGAL/kernel_basic.h>
#include <CGAL/enum.h>

#include <iostream>


namespace CGAL {

namespace internal {

template <class K>
typename K::Point_3
nearest_point_3(const typename K::Point_3& origin,
    	    const typename K::Tetrahedron_3& tetrahedron,
    	    const typename K::Point_3& bound,
    	    const K& k)
{
  typedef typename K::FT FT;
  typedef typename K::Point_3 Point_3;
  typedef typename K::Triangle_3 Triangle_3;
  typedef typename K::Tetrahedron_3 Tetrahedron_3;

  typename K::Compute_squared_distance_3 sq_distance =
    k.compute_squared_distance_3_object();
  typename K::Compare_squared_distance_3 compare_sq_distance =
    k.compare_squared_distance_3_object();
  
  // Return origin if origin lies inside tetrahedron
  if (do_intersect(origin,tetrahedron,k))
      return origin;

  // If origin is not in tetrahedron, compute nearest point for
  // each triangle and return the closest one.
  Point_3 closest_pt = nearest_point_3(origin, Triangle_3(tetrahedron[0], tetrahedron[1], tetrahedron[2]), bound);
  FT best_sq_dist = sq_distance(origin, closest_pt);

  Point_3 p = nearest_point_3(origin, Triangle_3(tetrahedron[0], tetrahedron[1], tetrahedron[3]), bound);
  FT sq_dist = sq_distance(origin,p);
  if (sq_dist < best_sq_dist)
  {
    best_sq_dist = sq_dist;
    closest_pt = p;
  }

  p = nearest_point_3(origin, Triangle_3(tetrahedron[0], tetrahedron[2], tetrahedron[3]), bound);
  sq_dist = sq_distance(origin,p);
  if (sq_dist < best_sq_dist)
  {
    best_sq_dist = sq_dist;
    closest_pt = p;
  }

  p = nearest_point_3(origin, Triangle_3(tetrahedron[1], tetrahedron[2], tetrahedron[3]), bound);
  sq_dist = sq_distance(origin,p);
  if (sq_dist < best_sq_dist)
  {
    best_sq_dist = sq_dist;
    closest_pt = p;
  }

  return closest_pt;
}

}

template <class K>
inline
Point_3<K>
nearest_point_3(const Point_3<K>& origin,
                const Tetrahedron_3<K>& triangle,
                const Point_3<K>& bound)
{
  return internal::nearest_point_3(origin, triangle, bound, K());
}

}  // end namespace CGAL


#endif 
