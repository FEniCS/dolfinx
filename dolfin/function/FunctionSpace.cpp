// Copyright (C) 2008-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristoffer Selim, 2008.
// Modified by Martin Alnes, 2008.
// Modified by Garth N. Wells, 2008-2009.
// Modified by Kent-Andre Mardal, 2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2008-09-11
// Last changed: 2009-09-16

#include <dolfin/main/MPI.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/IntersectionDetector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/la/GenericVector.h>
#include "Coefficient.h"
#include "FunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(boost::shared_ptr<const Mesh> mesh,
                             boost::shared_ptr<const FiniteElement> element,
                             boost::shared_ptr<const DofMap> dofmap)
  : _mesh(mesh), _element(element), _dofmap(dofmap),
    _restriction(static_cast<MeshFunction<bool>*>(0)),
    scratch(*element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(boost::shared_ptr<Mesh> mesh,
                             boost::shared_ptr<const FiniteElement> element,
                             boost::shared_ptr<const DofMap> dofmap)
  : _mesh(mesh), _element(element), _dofmap(dofmap),
    _restriction(static_cast<MeshFunction<bool>*>(0)),
    scratch(*element)
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
}
//-----------------------------------------------------------------------------
FunctionSpace::~FunctionSpace()
{
  // Do nothing
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
  return *this;
}
//-----------------------------------------------------------------------------
const Mesh& FunctionSpace::mesh() const
{
  assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
const FiniteElement& FunctionSpace::element() const
{
  assert(_element);
  return *_element;
}
//-----------------------------------------------------------------------------
const DofMap& FunctionSpace::dofmap() const
{
  assert(_dofmap);
  return *_dofmap;
}
//-----------------------------------------------------------------------------
dolfin::uint FunctionSpace::dim() const
{
  return dofmap().global_dimension();
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(GenericVector& coefficients,
                                const Coefficient& v, std::string meshes) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

  warning("FunctionSpace::interpolate requires revision."); 
  //if (meshes == "matching")
  //  assert(&v.function_space().mesh() == &mesh());
  //else if (meshes != "non-matching")
  //  error("Unknown mesh matching string %s in FunctionSpace::interpolate", meshes.c_str());

  // Initialize vector of coefficients
  coefficients.resize(_dofmap->global_dimension());
  coefficients.zero();

  // Iterate over mesh and interpolate on each cell
  UFCCell ufc_cell(*_mesh);
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Restrict function to cell
    v.restrict(scratch.coefficients, this->element(), *cell, ufc_cell, -1);

    // Tabulate dofs
    _dofmap->tabulate_dofs(scratch.dofs, ufc_cell, cell->index());

    // Copy dofs to vector
    coefficients.set(scratch.coefficients, _dofmap->local_dimension(ufc_cell), scratch.dofs);
  }

  // Finalise changes
  coefficients.apply();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace> FunctionSpace::operator[] (uint i) const
{
  std::vector<uint> component;
  component.push_back(i);
  return extract_sub_space(component);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace>
     FunctionSpace::extract_sub_space(const std::vector<uint>& component) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

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

  // Extract sub dofmap
  boost::shared_ptr<DofMap> dofmap(_dofmap->extract_sub_dofmap(component));

  // Create new sub space
  boost::shared_ptr<FunctionSpace> new_sub_space(new FunctionSpace(_mesh, element, dofmap));

  // Insert new sub space into cache
  subspaces.insert(std::pair<std::string, boost::shared_ptr<FunctionSpace> >(identifier.str(), new_sub_space));

  return new_sub_space;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace>
     FunctionSpace::collapse_sub_space(boost::shared_ptr<DofMap> dofmap) const
{
  boost::shared_ptr<FunctionSpace> collapsed_sub_space(new FunctionSpace(_mesh, _element, dofmap));
  return collapsed_sub_space;
}
//-----------------------------------------------------------------------------
void FunctionSpace::attach(MeshFunction<bool>& restriction)
{
  error("FunctionSpace::attach is not working. Please fix the dof map builder.");
  /*
  if (restriction.dim() == (*_mesh).topology().dim())
  {
    _restriction.reset(&restriction);
    //FIXME: hack to cast away the const
    const_cast<DofMap&>(*_dofmap).build(restriction);
  }
  */
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
void FunctionSpace::update()
{
  // FIXME: Ugly hack until we've figured out what the correct constness
  // FIXME: should be for DofMap, also affects generated code.
  const_cast<DofMap&>(*_dofmap).update();
}
//-----------------------------------------------------------------------------
