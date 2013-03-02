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
// Last changed: 2011-11-15

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/NoDeleter.h>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
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
    : _mesh(reference_to_no_delete_pointer(mesh)),
    _label(0),
    _use_labels(false),
    _kernel_type(kernel_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IntersectionOperator::IntersectionOperator(boost::shared_ptr<const Mesh> mesh,
                                           const std::string& kernel_type)
    : _mesh(mesh),
    _labels(new MeshFunction<std::size_t>()),
    _label(0),
    _use_labels(false),
    _kernel_type(kernel_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IntersectionOperator::IntersectionOperator(const MeshFunction<std::size_t>& labels,
					   std::size_t label,
					   const std::string& kernel_type)
  : _mesh(new Mesh()),
  _labels(reference_to_no_delete_pointer(labels)),
  _label(label),
  _use_labels(true),
  _kernel_type(kernel_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IntersectionOperator::IntersectionOperator(boost::shared_ptr<const MeshFunction<std::size_t> > labels,
					   std::size_t label,
					   const std::string& kernel_type)
  : _mesh(new Mesh()),
  _labels(labels),
  _label(label),
  _use_labels(true),
  _kernel_type(kernel_type)
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
                                                    std::set<std::size_t>& ids_result) const
{
  rImpl().all_intersected_entities(point, ids_result);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const std::vector<Point>& points,
                                                    std::set<std::size_t>& ids_result) const
{
  rImpl().all_intersected_entities(points, ids_result);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const MeshEntity & entity,
						    std::vector<std::size_t>& ids_result) const
{
  rImpl().all_intersected_entities(entity,ids_result);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const std::vector<MeshEntity>& entities,
						    std::set<std::size_t>& ids_result) const
{
  rImpl().all_intersected_entities(entities, ids_result);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::all_intersected_entities(const Mesh& another_mesh,
                                                    std::set<std::size_t>& ids_result) const
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
std::size_t IntersectionOperator::closest_cell(const Point& point) const
{
  return rImpl().closest_cell(point);
}
//-----------------------------------------------------------------------------
std::pair<Point, std::size_t>
IntersectionOperator::closest_point_and_cell(const Point& point) const
{
  return rImpl().closest_point_and_cell(point);
}
//-----------------------------------------------------------------------------
double IntersectionOperator::distance(const Point & point) const
{
  return rImpl().distance(point);
}
//-----------------------------------------------------------------------------
void IntersectionOperator::reset_kernel(const std::string& kernel_type)
{
  _pImpl.reset(create_intersection_operator(kernel_type));
}
//-----------------------------------------------------------------------------
void IntersectionOperator::clear()
{
  _pImpl.reset();
}
//-----------------------------------------------------------------------------
const Mesh& IntersectionOperator::mesh() const
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
const IntersectionOperatorImplementation& IntersectionOperator::rImpl() const
{
  if (!_pImpl)
    _pImpl.reset(const_cast<IntersectionOperator *>(this)->create_intersection_operator(_kernel_type));
  return *_pImpl;
}
//-----------------------------------------------------------------------------
#ifdef HAS_CGAL
IntersectionOperatorImplementation*
    IntersectionOperator::create_intersection_operator(
				    const std::string& kernel_type = "SimpleCartesian")
{
  if (!_use_labels)
  {
    if (kernel_type == "ExactPredicates")
    {
      switch(_mesh->type().cell_type())
      {
        case CellType::point      : return new IntersectionOperatorImplementation_d<PointCell, EPICK>(_mesh);
        case CellType::interval   : return new IntersectionOperatorImplementation_d<IntervalCell, EPICK>(_mesh);
        case CellType::triangle   : return new IntersectionOperatorImplementation_d<TriangleCell, EPICK>(_mesh);
        case CellType::tetrahedron: return new IntersectionOperatorImplementation_d<TetrahedronCell, EPICK>(_mesh);
        default: dolfin_error("IntersectionOperator.cpp",
                            "create intersection operator",
                            "Cell type of mesh is not known. Allowed cell types are point, interval, triangle and tetrahedron");
		 return 0;
      }
    }
    else  // Default is SimpleCartesion
    {
      switch( _mesh->type().cell_type())
      {
        case CellType::point      : return new IntersectionOperatorImplementation_d< PointCell, SCK  >(_mesh);
        case CellType::interval   : return new IntersectionOperatorImplementation_d< IntervalCell, SCK  >(_mesh);
        case CellType::triangle   : return new IntersectionOperatorImplementation_d< TriangleCell, SCK >(_mesh);
        case CellType::tetrahedron: return new IntersectionOperatorImplementation_d< TetrahedronCell, SCK  >(_mesh);
        default:
        {
          dolfin_error("IntersectionOperator.cpp",
                       "create intersection operator",
                       "Cell type of mesh is not known. Allowed cell types are point, interval, triangle and tetrahedron");
        }
      }
    }
  }
  else
  {
    if (kernel_type == "ExactPredicates")
    {
      switch(_labels->dim())
      {
        case CellType::point      : return new IntersectionOperatorImplementation_d<PointCell, EPICK>(*_labels, _label);
        case CellType::interval   : return new IntersectionOperatorImplementation_d<IntervalCell, EPICK>(*_labels, _label);
        case CellType::triangle   : return new IntersectionOperatorImplementation_d<TriangleCell, EPICK>(*_labels, _label);
        case CellType::tetrahedron: return new IntersectionOperatorImplementation_d<TetrahedronCell, EPICK>(*_labels, _label);
        default: dolfin_error("IntersectionOperator.cpp",
                            "create intersection operator",
                            "Cell type of mesh is not known. Allowed cell types are point, interval, triangle and tetrahedron");
        return 0;
      }
    }
    else  // Default is SimpleCartesion
    {
      switch(_labels->dim())
      {
        case CellType::point      : return new IntersectionOperatorImplementation_d< PointCell, SCK  >(*_labels, _label);
        case CellType::interval   : return new IntersectionOperatorImplementation_d< IntervalCell, SCK  >(*_labels, _label);
        case CellType::triangle   : return new IntersectionOperatorImplementation_d< TriangleCell, SCK >(*_labels, _label);
        case CellType::tetrahedron: return new IntersectionOperatorImplementation_d< TetrahedronCell, SCK  >(*_labels, _label);
        default: dolfin_error("IntersectionOperator.cpp",
                            "create intersection operator",
                            "Cell type of mesh is not known. Allowed cell types are point, interval, triangle and tetrahedron");
  		 return 0;
      }
    }
  }
}

#else
//If CGAL support is not available, throw an exception.
IntersectionOperatorImplementation*
    IntersectionOperator::create_intersection_operator(
                                               const std::string & kernel_type = "SimpleCartesian")
{
  dolfin_error("IntersectionOperator.cpp",
               "create intersection operator",
               "IntersectionOperatorImplementation is not available, DOLFIN has been compiled without CGAL");

  // Use to member variable to avoid fuss compiler warnings
  dolfin_assert(_label);
  dolfin_assert(_use_labels);

  return 0;
}
#endif
//-----------------------------------------------------------------------------
