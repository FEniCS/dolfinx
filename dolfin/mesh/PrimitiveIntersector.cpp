// =====================================================================================
//
// Copyright (C) 2010-02-09  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-02-09
// Last changed: 2010-02-11
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#include "MeshEntity.h"
#include "PrimitiveIntersector.h"

using namespace dolfin;

#ifdef HAS_CGAL
#include "cgal_includes.h"

//-----------------------------------------------------------------------------
bool PrimitiveIntersector::do_intersect(const MeshEntity & entity_1, const MeshEntity & entity_2)
{
  return PrimitiveIntersector::do_intersect_with_kernel<SCK>(entity_1, entity_2);
}

bool PrimitiveIntersector::do_intersect_exact(const MeshEntity & entity_1, const MeshEntity & entity_2)
{

  return PrimitiveIntersector::do_intersect_with_kernel<EPICK>(entity_1, entity_2);
}
//-----------------------------------------------------------------------------
template <typename K, typename T, typename U >
bool PrimitiveIntersector::do_intersect_with_kernel(const T & entity_1, const U & entity_2)
{
  return CGAL::do_intersect(entity_1, entity_2);
}
//-----------------------------------------------------------------------------
template<typename K, typename T >
bool PrimitiveIntersector::do_intersect_with_kernel(const T & entity_1, const MeshEntity & entity_2)
{
  switch(entity_2.dim())
  {
    case 0 :	return do_intersect_with_kernel<K>(PrimitiveTraits<PointCell,K>::datum(entity_2), entity_1);
    case 1 :	return do_intersect_with_kernel<K>(PrimitiveTraits<IntervalCell,K>::datum(entity_2), entity_1);
    case 2 :	return do_intersect_with_kernel<K>(PrimitiveTraits<TriangleCell,K>::datum(entity_2),  entity_1);
    case 3 :	return do_intersect_with_kernel<K>(PrimitiveTraits<TetrahedronCell,K>::datum(entity_2), entity_1);
    default:  error("DOLFIN PrimitiveIntersector::do_intersect: \n Dimension of  is not known."); return false;
  }
}

#else

#include <dolfin/log/log.h>

bool PrimitiveIntersector::do_intersect(const MeshEntity & entity_1, const MeshEntity & entity_2)
{
  warning("DOLFIN has been compiled without CGAL support");
  dolfin_not_implemented();
  return false;
}
//-----------------------------------------------------------------------------
bool PrimitiveIntersector::do_intersect_exact(const MeshEntity & entity_1, const MeshEntity & entity_2)
{
  warning("DOLFIN has been compiled without CGAL support");
  dolfin_not_implemented();
  return false;
}

#endif
