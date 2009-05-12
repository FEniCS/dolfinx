// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristoffer Selim, 2008.
// Modified by Martin Alnes, 2008.
// Modified by Garth N. Wells, 2008.
// Modified by Kent-Andre Mardal, 2009.
//
// First added:  2008-09-11
// Last changed: 2009-01-06

#include <dolfin/fem/UFC.h>
#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/IntersectionDetector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/la/GenericVector.h>
#include "Function.h"
#include "FunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(const Mesh& mesh,
                             const FiniteElement &element,
                             const DofMap& dofmap)
  : _mesh(reference_to_no_delete_pointer(mesh)),
    _element(reference_to_no_delete_pointer(element)),
    _dofmap(reference_to_no_delete_pointer(dofmap)),
    _restriction(static_cast<MeshFunction<bool>*>(0)),
    scratch(element), intersection_detector(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(boost::shared_ptr<const Mesh> mesh,
                             boost::shared_ptr<const FiniteElement> element,
                             boost::shared_ptr<const DofMap> dofmap)
  : _mesh(mesh), _element(element), _dofmap(dofmap),
    _restriction(static_cast<MeshFunction<bool>*>(0)),
    scratch(*element), intersection_detector(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(const FunctionSpace& V)
{
  // Assign data (will be shared)
  _mesh    = V._mesh;
  _element = V._element;
  _dofmap  = V._dofmap;
  _restriction = V._restriction;

  // Reinitialize scratch space and intersection detector
  scratch.init(*_element);
  intersection_detector = 0;
}
//-----------------------------------------------------------------------------
FunctionSpace::~FunctionSpace()
{
  delete intersection_detector;
}
//-----------------------------------------------------------------------------
const FunctionSpace& FunctionSpace::operator= (const FunctionSpace& V)
{
  // Assign data (will be shared)
  _mesh    = V._mesh;
  _element = V._element;
  _dofmap  = V._dofmap;
  _restriction = V._restriction;

  // Reinitialize scratch space and intersection detector
  scratch.init(*_element);
  if (intersection_detector)
  {
    delete intersection_detector;
    intersection_detector = 0;
  }
  return *this;
}
//-----------------------------------------------------------------------------
const Mesh& FunctionSpace::mesh() const
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
const FiniteElement& FunctionSpace::element() const
{
  dolfin_assert(_element);
  return *_element;
}
//-----------------------------------------------------------------------------
const DofMap& FunctionSpace::dofmap() const
{
  dolfin_assert(_dofmap);
  return *_dofmap;
}
//-----------------------------------------------------------------------------
dolfin::uint FunctionSpace::dim() const
{
  return dofmap().global_dimension();
}
//-----------------------------------------------------------------------------
void FunctionSpace::eval(double* values,
                         const double* x,
                         const Function& v) const
{
  dolfin_assert(values);
  dolfin_assert(x);
  dolfin_assert(v.in(*this));
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Initialize intersection detector if not done before
  if (!intersection_detector)
    intersection_detector = new IntersectionDetector(*_mesh);

  // Find the cell that contains x
  Point point(_mesh->geometry().dim(), x);
  std::vector<uint> cells;
  intersection_detector->intersection(point, cells);
  if (cells.size() < 1)
    error("Unable to evaluate function at given point (not inside domain).");
  Cell cell(*_mesh, cells[0]);
  UFCCell ufc_cell(cell);

  // Evaluate at point
  eval(values, x, v, ufc_cell, cell.index());
}
//-----------------------------------------------------------------------------
void FunctionSpace::eval(double* values,
                         const double* x,
                         const Function& v,
                         const ufc::cell& ufc_cell,
                         uint cell_index) const
{
  dolfin_assert(values);
  dolfin_assert(x);
  dolfin_assert(v.in(*this));
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Interpolate function to cell
  v.interpolate(scratch.coefficients, *this, ufc_cell, cell_index);

  // Compute linear combination
  for (uint j = 0; j < scratch.size; j++)
    values[j] = 0.0;
  for (uint i = 0; i < _element->space_dimension(); i++)
  {
    _element->evaluate_basis(i, scratch.values, x, ufc_cell);
    for (uint j = 0; j < scratch.size; j++)
      values[j] += scratch.coefficients[i] * scratch.values[j];
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(GenericVector& coefficients,
                                const Function& v) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);
  dolfin_assert(&v.function_space().mesh() == &mesh());

  // Initialize vector of coefficients
  coefficients.resize(_dofmap->global_dimension());
  coefficients.zero();

  // Iterate over mesh and interpolate on each cell
  UFCCell ufc_cell(*_mesh);
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Interpolate on cell
    v.interpolate(scratch.coefficients, *this, ufc_cell, cell->index());

    // Tabulate dofs
    _dofmap->tabulate_dofs(scratch.dofs, ufc_cell, cell->index());

    // Copy dofs to vector
    coefficients.set(scratch.coefficients, _dofmap->local_dimension(ufc_cell), scratch.dofs);
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(double* vertex_values,
                                const Function& v) const
{
  dolfin_assert(vertex_values);
  dolfin_assert(v.in(*this));
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Local data for interpolation on each cell
  const uint num_cell_vertices = _mesh->type().num_vertices(_mesh->topology().dim());
  double* local_vertex_values = new double[scratch.size*num_cell_vertices];

  // Interpolate vertex values on each cell (using latest value if not continuous)
  UFCCell ufc_cell(*_mesh);
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Tabulate dofs
    _dofmap->tabulate_dofs(scratch.dofs, ufc_cell, cell->index());

    // Pick values from global vector
    v.interpolate(scratch.coefficients, ufc_cell, cell->index());

    // Interpolate values at the vertices
    _element->interpolate_vertex_values(local_vertex_values, scratch.coefficients, ufc_cell);

    // Copy values to array of vertex values
    for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
    {
      for (uint i = 0; i < scratch.size; ++i)
      {
        const uint local_index  = vertex.pos()*scratch.size + i;
        const uint global_index = i*_mesh->num_vertices() + vertex->index();
        vertex_values[global_index] = local_vertex_values[local_index];
      }
    }
  }

  // Delete local data
  delete [] local_vertex_values;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace> FunctionSpace::extract_sub_space(const std::vector<uint>& component) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Create unique identifier string for sub space
  std::ostringstream identifier;
  for (uint i = 0; i < component.size(); ++i)
    identifier << component[i] << ".";

  // Check if sub space is aleady in the cache
  std::map<std::string, boost::shared_ptr<FunctionSpace> >::iterator subspace;
  subspace = subspaces.find(identifier.str());
  if (subspace != subspaces.end())
    return subspace->second;

  // Extract sub element
  boost::shared_ptr<const FiniteElement> element(_element->extract_sub_element(component));

  // Extract sub dofmap and offset
  uint offset = 0;
  boost::shared_ptr<DofMap> dofmap(_dofmap->extract_sub_dofmap(component, offset, *_mesh));

  // Create new sub space
  boost::shared_ptr<FunctionSpace> new_sub_space(new FunctionSpace(_mesh, element, dofmap));

  // Insert new sub space into cache
  subspaces.insert(std::pair<std::string, boost::shared_ptr<FunctionSpace> >(identifier.str(), new_sub_space));

  return new_sub_space;
}
//-----------------------------------------------------------------------------
void FunctionSpace:: attach(MeshFunction<bool>& restriction)
{
  if (restriction.dim() == (*_mesh).topology().dim())
  {
    _restriction.reset(&restriction);
    //FIXME: hack to cast away the const
    const_cast<DofMap&>(*_dofmap).build(*_mesh, *_element, restriction);
  }
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace> FunctionSpace::restriction(MeshFunction<bool>& restriction)
{
  boost::shared_ptr<FunctionSpace> function_space(new FunctionSpace(_mesh, _element, _dofmap));
  function_space->attach(restriction);
  return function_space;
}
//-----------------------------------------------------------------------------
FunctionSpace::Scratch::Scratch(const FiniteElement& element)
  : size(0), dofs(0), coefficients(0), values(0)
{
  init(element);
}
//-----------------------------------------------------------------------------
FunctionSpace::Scratch::Scratch()
  : size(0), dofs(0), coefficients(0), values(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::Scratch::~Scratch()
{
  delete [] dofs;
  delete [] coefficients;
  delete [] values;
}
//-----------------------------------------------------------------------------
void FunctionSpace::Scratch::init(const FiniteElement& element)
{
  // Compute size of value (number of entries in tensor value)
  size = 1;
  for (uint i = 0; i < element.value_rank(); i++)
    size *= element.value_dimension(i);

  // Initialize local array for mapping of dofs
  delete [] dofs;
  dofs = new uint[element.space_dimension()];
  for (uint i = 0; i < element.space_dimension(); i++)
    dofs[i] = 0;

  // Initialize local array for expansion coefficients
  delete [] coefficients;
  coefficients = new double[element.space_dimension()];
  for (uint i = 0; i < element.space_dimension(); i++)
    coefficients[i] = 0.0;

  // Initialize local array for values
  delete [] values;
  values = new double[size];
  for (uint i = 0; i < size; i++)
    values[i] = 0.0;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::is_inside_restriction(uint c) const
{
  if (_restriction)
    return _restriction->get(c);
  else
    return true;
}
//-----------------------------------------------------------------------------
