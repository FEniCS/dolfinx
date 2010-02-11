// =====================================================================================
//
// Copyright (C) 2010-02-05  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-02-05
// Last changed: 2010-02-10
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#ifdef HAS_CGAL

#include "added_intersection_3.h" //additional intersection functionality, *Must* include before the AABB_tree!
//#include <CGAL/intersections.h>

#include <CGAL/AABB_tree.h> // *Must* be inserted before kernel!
#include <CGAL/AABB_traits.h>

#include <CGAL/Simple_cartesian.h> 
#include "Triangle_3_Tetrahedron_3_do_intersect_SCK.h" //template specialization for Simple_cartesian kernel

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Bbox_3.h>
#include <CGAL/Point_3.h>

#include "PrimitiveTraits.h"

#include "MeshPrimitive.h"

typedef CGAL::Simple_cartesian<double> SCK;
typedef CGAL::Exact_predicates_inexact_constructions_kernel EPICK;

#endif
