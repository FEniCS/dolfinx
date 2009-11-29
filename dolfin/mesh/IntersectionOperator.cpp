// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-01
// Last changed: 2009-11-28

#include <algorithm>
#include <map>
#include <string>

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
#include "MeshPrimitive.h"
#include "Primitive_Traits.h"

using namespace dolfin;

#ifdef HAS_CGAL

IntersectionOperator::IntersectionOperator(const Mesh & mesh, const std::string & kernel_type)
    : _mesh(reference_to_no_delete_pointer(mesh)),_kernel_type(kernel_type) {}

IntersectionOperator::IntersectionOperator(boost::shared_ptr<const Mesh> mesh, const std::string & kernel_type)
    : _mesh(mesh), _kernel_type(kernel_type) {}

IntersectionOperator::~IntersectionOperator() {}

void IntersectionOperator::all_intersected_entities(const Point & point, uint_set & ids_result) const
{ rImpl().all_intersected_entities(point,ids_result);}

void IntersectionOperator::all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) const
{ rImpl().all_intersected_entities(points,ids_result);}

void IntersectionOperator::all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) const
{ rImpl().all_intersected_entities(another_mesh, ids_result);}

int IntersectionOperator::any_intersected_entity(const Point & point) const
{ return rImpl().any_intersected_entity(point);}

void IntersectionOperator::reset_kernel(const std::string & kernel_type) 
{ _pImpl.reset(create_intersection_operator(_mesh,kernel_type)); }

void IntersectionOperator::clear() 
{ _pImpl.reset(); }

const Mesh& IntersectionOperator::mesh() const
{
  assert(_mesh);
  return *_mesh;
}

boost::shared_ptr<const Mesh> IntersectionOperator::mesh_ptr() 
{ 
  assert(_mesh);
  return _mesh;
}

IntersectionOperatorImplementation * IntersectionOperator::create_intersection_operator(boost::shared_ptr<const Mesh> mesh, const std::string & kernel_type = "SimpleCartesian")
{
  if (kernel_type == "ExactPredicates")
  {
    switch( mesh->type().cell_type())
    {
      case CellType::point        : return new IntersectionOperatorImplementation_d< Primitive_Traits<PointCell, EPICK> >(mesh);
      case CellType::interval     : return new IntersectionOperatorImplementation_d< Primitive_Traits<IntervalCell, EPICK> >(mesh);
      case CellType::triangle     : return new IntersectionOperatorImplementation_d< Primitive_Traits<TriangleCell, EPICK> >(mesh); 
      case CellType::tetrahedron  : return new IntersectionOperatorImplementation_d< Primitive_Traits<TetrahedronCell, EPICK> >(mesh);
      default: error("DOLFIN IntersectionOperator::create_intersection_operator: \n Mesh  CellType is not known."); return 0;
    }
  }
  //default is SimpleCartesion
  else 
  {
//    if (kernel_type != "SimpleCartesian")
//      warning("Type %s of geometry kernel is not  known. Falling back to SimpleCartesian.",kernel_type);
    switch( mesh->type().cell_type())
    {
      case CellType::point        : return new IntersectionOperatorImplementation_d< Primitive_Traits<PointCell, SCK > >(mesh);
      case CellType::interval     : return new IntersectionOperatorImplementation_d< Primitive_Traits<IntervalCell, SCK > >(mesh);
      case CellType::triangle     : return new IntersectionOperatorImplementation_d< Primitive_Traits<TriangleCell, SCK> >(mesh); 
      case CellType::tetrahedron  : return new IntersectionOperatorImplementation_d< Primitive_Traits<TetrahedronCell, SCK > >(mesh);
      default: error("DOLFIN IntersectionOperator::create_intersection_operator: \n Mesh  CellType is not known."); return 0;
    }
  }
}

const IntersectionOperatorImplementation& IntersectionOperator::rImpl() const
{
  if (!_pImpl)
    _pImpl.reset(const_cast<IntersectionOperator *>(this)->create_intersection_operator(_mesh,_kernel_type));
  return *_pImpl;
}

#else

IntersectionOperator::IntersectionOperator(const Mesh & _mesh)
{
  error("DOLFIN has been compiled without CGAL, IntersectionOperator is not available.");
}

IntersectionOperator::IntersectionOperator(boost::shared_ptr<const Mesh> _mesh, std::string &)
{
  error("DOLFIN has been compiled without CGAL, IntersectionOperator is not available.");
}

IntersectionOperator::~IntersectionOperator() {}

void IntersectionOperator::all_intersected_entities(const Point & point, uint_set & ids_result) {}

void IntersectionOperator::all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) {}

void IntersectionOperator::all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) {}

int IntersectionOperator::any_intersected_entity(const Point & point) {return -1;}

void IntersectionOperator::clear() {}

const Mesh& IntersectionOperator::mesh() const
{
  assert(_mesh);
  return *_mesh;
}

boost::shared_ptr<Mesh> IntersectionOperator::mesh_ptr() 
{ 
  assert(_mesh);
  return _mesh;
}

void IntersectionOperator::reset_kernel(const std::string & kernel_type) {}
void IntersectionOperator::reset() {}

#endif
