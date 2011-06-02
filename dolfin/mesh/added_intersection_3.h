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
// Last changed: 2010-02-08

#ifndef  ADDED_INTERSECTION_3_INC
#define  ADDED_INTERSECTION_3_INC

//#ifdef HAS_CGAL

#include "Point_3_Point_3_intersection.h"
#include "Point_3_Segment_3_intersection.h"
//#include "Point_3_Triangle_3_intersection.h" //already defined in original
//CGAL, but not with p.has_on(pt) function??
#include "Point_3_Tetrahedron_3_intersection.h"
#include "Point_3_Iso_Cuboid_3_intersection.h"
#include "Point_3_Line_3_intersection.h"
#include "Point_3_Ray_3_intersection.h"
#include "Point_3_Bbox_3_intersection.h"
#include "Segment_3_Segment_3_intersection.h"
#include "Segment_3_Tetrahedron_3_intersection.h"
#include "Tetrahedron_3_Tetrahedron_3_intersection.h"
//#include "Triangle_3_Tetrahedron_3_do_intersect_SCK.h"
#include "Tetrahedron_3_Bbox_3_intersection.h"

//#endif

#endif   /* ----- #ifndef ADDED_INTERSECTION_3_INC  ----- */
