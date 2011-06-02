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
// First added:  2009-09-01
// Last changed: 2010-03-03

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>
#include <dolfin/common/NoDeleter.h>

#include "Mesh.h"
#include "Edge.h"
#include "Facet.h"
#include "Vertex.h"
#include "Cell.h"
#include "IntersectionOperator.h"
#include "IntersectionOperatorImplementation.h"

#ifdef HAS_CGAL
#include "MeshPrimitive.h"
#include "PrimitiveTraits.h"
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
IntersectionOperator::IntersectionOperator(const Mesh& mesh,
                                           const std::string& kernel_type)
    : _mesh(reference_to_no_delete_pointer(mesh)),_kernel_type(kernel_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IntersectionOperator::IntersectionOperator(boost::shared_ptr<const Mesh> mesh,
                                           const std::string& kernel_type)
    : _mesh(mesh), _kernel_type(kernel_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IntersectionOperator::~IntersectionOperator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const Point& point,
                                                    uint_set& ids_result) const
{
  rImpl().all_intersected_entities(point, ids_result);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const std::vector<Point>& points,
                                                   uint_set& ids_result) const
{
  rImpl().all_intersected_entities(points, ids_result);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const MeshEntity & entity,
						    std::vector<uint> & ids_result) const
{
  rImpl().all_intersected_entities(entity,ids_result);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const std::vector<MeshEntity>& entities,
						    uint_set& ids_result) const
{
  rImpl().all_intersected_entities(entities, ids_result);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const Mesh& another_mesh,
                                                    uint_set& ids_result) const
{
  rImpl().all_intersected_entities(another_mesh, ids_result);
}
//-----------------------------------------------------------------------------
int IntersectionOperator::any_intersected_entity(const Point& point) const
{
  return rImpl().any_intersected_entity(point);
}
//-----------------------------------------------------------------------------
Point IntersectionOperator::closest_point(const Point& point) const
{
  return rImpl().closest_point(point);
}
//-----------------------------------------------------------------------------
dolfin::uint IntersectionOperator::closest_cell(const Point& point) const
{
  return rImpl().closest_cell(point);
}
//-----------------------------------------------------------------------------
std::pair<Point,dolfin::uint> IntersectionOperator::closest_point_and_cell(const Point& point) const
{
  return rImpl().closest_point_and_cell(point);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::reset_kernel(const std::string& kernel_type)
{
  _pImpl.reset(create_intersection_operator(_mesh, kernel_type));
}
//-----------------------------------------------------------------------------
void IntersectionOperator::clear()
{
  _pImpl.reset();
}
//-----------------------------------------------------------------------------
const Mesh& IntersectionOperator::mesh() const
{
  assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
const IntersectionOperatorImplementation& IntersectionOperator::rImpl() const
{
  if (!_pImpl)
    _pImpl.reset(const_cast<IntersectionOperator *>(this)->create_intersection_operator(_mesh,_kernel_type));
  return *_pImpl;
}
//-----------------------------------------------------------------------------
#ifdef HAS_CGAL

IntersectionOperatorImplementation*
    IntersectionOperator::create_intersection_operator(boost::shared_ptr<const Mesh> mesh,
						       const std::string& kernel_type = "SimpleCartesian")
{
  if (kernel_type == "ExactPredicates")
  {
    switch( mesh->type().cell_type())
    {
      case CellType::point      : return new IntersectionOperatorImplementation_d<PointCell, EPICK>(mesh);
      case CellType::interval   : return new IntersectionOperatorImplementation_d<IntervalCell, EPICK>(mesh);
      case CellType::triangle   : return new IntersectionOperatorImplementation_d<TriangleCell, EPICK>(mesh);
      case CellType::tetrahedron: return new IntersectionOperatorImplementation_d<TetrahedronCell, EPICK>(mesh);
      default: error("DOLFIN IntersectionOperator::create_intersection_operator: \n Mesh  CellType is not known.");
      return 0;
    }
  }
  else  // Default is SimpleCartesion
  {
    switch( mesh->type().cell_type())
    {
      case CellType::point      : return new IntersectionOperatorImplementation_d< PointCell, SCK  >(mesh);
      case CellType::interval   : return new IntersectionOperatorImplementation_d< IntervalCell, SCK  >(mesh);
      case CellType::triangle   : return new IntersectionOperatorImplementation_d< TriangleCell, SCK >(mesh);
      case CellType::tetrahedron: return new IntersectionOperatorImplementation_d< TetrahedronCell, SCK  >(mesh);
      default: error("DOLFIN IntersectionOperator::create_intersection_operator: \n Mesh  CellType is not known.");
      return 0;
    }
  }
}

#else
//If CGAL support is not available, throw an exception.
IntersectionOperatorImplementation*
    IntersectionOperator::create_intersection_operator(boost::shared_ptr<const Mesh> mesh,
                                               const std::string & kernel_type = "SimpleCartesian")
{
  error("DOLFIN has been compiled without CGAL, IntersectionOperatorImplementation is not available.");
  return 0;
}
#endif
//-----------------------------------------------------------------------------

