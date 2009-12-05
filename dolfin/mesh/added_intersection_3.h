// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2009-12-05

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
