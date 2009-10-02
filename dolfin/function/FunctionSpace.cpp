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
// Last changed: 2009-10-02

#include <boost/scoped_array.hpp>

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
    _restriction(static_cast<MeshFunction<bool>*>(0))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(boost::shared_ptr<Mesh> mesh,
                             boost::shared_ptr<const FiniteElement> element,
                             boost::shared_ptr<const DofMap> dofmap)
  : _mesh(mesh), _element(element), _dofmap(dofmap),
    _restriction(static_cast<MeshFunction<bool>*>(0))
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
void FunctionSpace::interpolate(GenericVector& expansion_coefficients,
                                const Coefficient& v) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

  // Initialize vector of expansion coefficients
  expansion_coefficients.resize(_dofmap->global_dimension());
  expansion_coefficients.zero();

  // Initialize local arrays
  const uint max_local_dimension = _dofmap->max_local_dimension();
  boost::scoped_array<double> cell_coefficients(new double[max_local_dimension]);
  boost::scoped_array<uint> cell_dofs(new uint[max_local_dimension]);

  // Iterate over mesh and interpolate on each cell
  UFCCell ufc_cell(*_mesh);
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Restrict function to cell
    v.restrict(cell_coefficients.get(), this->element(), *cell, ufc_cell);

    // Tabulate dofs
    _dofmap->tabulate_dofs(cell_dofs.get(), ufc_cell, cell->index());

    // Copy dofs to vector
    expansion_coefficients.set(cell_coefficients.get(),
                               _dofmap->local_dimension(ufc_cell),
                               cell_dofs.get());
  }

  // Finalise changes
  expansion_coefficients.apply();
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
bool FunctionSpace::has_element(const FiniteElement& element,
                                const Cell& cell) const
{
  assert(_mesh);
  assert(_element);

  return &element == &*_element && &cell.mesh() == &*_mesh;
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
