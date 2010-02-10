// =====================================================================================
//
// Copyright (C) 2010-02-09  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-02-09
// Last changed: 2010-02-10
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#include "MeshEntity.h"
#include "PrimitiveIntersector.h"
#include "cgal_includes.h"

using namespace dolfin;

bool PrimitiveIntersector::do_intersect(const MeshEntity & entity_1, const MeshEntity & entity_2)
{
  return PrimitiveIntersector::do_intersect_with_kernel<SCK>(entity_1, entity_2);
}

bool PrimitiveIntersector::do_intersect_exact(const MeshEntity & entity_1, const MeshEntity & entity_2)
{

  return PrimitiveIntersector::do_intersect_with_kernel<EPICK>(entity_1, entity_2);
}

template <typename Kernel>
bool PrimitiveIntersector::do_intersect_with_kernel(const MeshEntity & entity_1, const MeshEntity & entity_2)
{
  switch(entity_1.dim())
  {
    case 0 :	return PrimitiveIntersector::_do_intersect(PrimitiveTraits<PointCell,Kernel>::datum(entity_1), entity_2);
    case 1 :	return PrimitiveIntersector::_do_intersect(PrimitiveTraits<IntervalCell,Kernel>::datum(entity_1), entity_2);
    case 2 :	return PrimitiveIntersector::_do_intersect(PrimitiveTraits<TriangleCell,Kernel>::datum(entity_1), entity_2);
    case 3 :	return PrimitiveIntersector::_do_intersect(PrimitiveTraits<TetrahedronCell,Kernel>::datum(entity_1), entity_2);
    default:  error("DOLFIN PrimitiveIntersector::do_intersect_with_kernel: \n Dimension of  is not known."); return false;
  }
}

template<typename T>
bool PrimitiveIntersector::_do_intersect(const T & entity_1, const MeshEntity & entity_2)
{
  typedef typename T::R K;
  switch(entity_2.dim())
  {
    case 0 :	return CGAL::do_intersect(entity_1, PrimitiveTraits<PointCell,K>::datum(entity_2));
    case 1 :	return CGAL::do_intersect(entity_1, PrimitiveTraits<IntervalCell,K>::datum(entity_2));
    case 2 :	return CGAL::do_intersect(entity_1, PrimitiveTraits<TriangleCell,K>::datum(entity_2));
    case 3 :	return CGAL::do_intersect(entity_1, PrimitiveTraits<TetrahedronCell,K>::datum(entity_2));
    default:  error("DOLFIN PrimitiveIntersector::_do_intersect: \n Dimension of  is not known."); return false;
  }
}
