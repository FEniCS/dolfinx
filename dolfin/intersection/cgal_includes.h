// Copyright (C) 2010 Andre Massing
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
// Modified by Anders Logg 2011
//
// First added:  2010-02-05
// Last changed: 2013-04-22

#ifdef HAS_CGAL

#include "added_intersection_3.h" //  Additional intersection functionality, *must* include before the AABB_tree!
#include "nearest_point_tetrahedron_3.h"
#include <CGAL/AABB_tree.h>       // *Must* be inserted before kernel!
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
