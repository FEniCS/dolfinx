// Copyright (C) 2013-2016 Anders Logg
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
// First added:  2013-08-05
// Last changed: 2016-03-02

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include <dolfin/fem/MultiMeshDofMap.h>
#include "FunctionSpace.h"
#include "MultiMeshFunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshFunctionSpace::MultiMeshFunctionSpace(std::shared_ptr<const MultiMesh> multimesh)
  : _multimesh(multimesh),
    _dofmap(new MultiMeshDofMap())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MultiMeshFunctionSpace::~MultiMeshFunctionSpace()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t MultiMeshFunctionSpace::dim() const
{
  dolfin_assert(_dofmap);
  return _dofmap->global_dimension();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MultiMesh> MultiMeshFunctionSpace::multimesh() const
{
  dolfin_assert(_multimesh);
  return _multimesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const MultiMeshDofMap> MultiMeshFunctionSpace::dofmap() const
{
  dolfin_assert(_dofmap);
  return _dofmap;
}
//-----------------------------------------------------------------------------
std::size_t MultiMeshFunctionSpace::num_parts() const
{
  return _function_spaces.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace>
MultiMeshFunctionSpace::part(std::size_t i) const
{
  dolfin_assert(i < _function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace>
MultiMeshFunctionSpace::view(std::size_t i) const
{
  dolfin_assert(i < _function_space_views.size());
  return _function_space_views[i];
}
//-----------------------------------------------------------------------------
void
MultiMeshFunctionSpace::add(std::shared_ptr<const FunctionSpace> function_space)
{
  _function_spaces.push_back(function_space);
  log(PROGRESS, "Added function space to multimesh space; space has %d part(s).",
      _function_spaces.size());
}
//-----------------------------------------------------------------------------
void MultiMeshFunctionSpace::build()
{
  begin(PROGRESS, "Building multimesh function space.");

  // Build dofmap using empty list of offsets (will be computed)
  std::vector<dolfin::la_index> offsets;
  _build_dofmap(offsets);

  // Build views
  _build_views();

  end();
}
//-----------------------------------------------------------------------------
void MultiMeshFunctionSpace::build(const std::vector<dolfin::la_index>& offsets)
{
  begin(PROGRESS, "Building multimesh subspace.");

  // Build dofmap
  _build_dofmap(offsets);

  // Build views
  _build_views();

  end();
}
//-----------------------------------------------------------------------------
void MultiMeshFunctionSpace::_build_dofmap(const std::vector<dolfin::la_index>& offsets)
{
  begin(PROGRESS, "Building multimesh dofmap.");

  // Clear dofmap
  dolfin_assert(_dofmap);
  _dofmap->clear();

  // Add dofmap for each part
  for (std::size_t i = 0; i < num_parts(); i++)
    _dofmap->add(_function_spaces[i]->dofmap());

  // Call function to build dofmap
  _dofmap->build(*this, offsets);

  end();
}
//-----------------------------------------------------------------------------
void MultiMeshFunctionSpace::_build_views()
{
  // Clear old views
  _function_space_views.clear();

  // Iterate over parts
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    // Extract data
    auto mesh    = this->part(i)->mesh();
    auto element = this->part(i)->element();
    auto dofmap  = this->dofmap()->part(i);

    // Create function space
    std::shared_ptr<const FunctionSpace> V(new FunctionSpace(mesh,
                                                             element,
                                                             dofmap));

    // Add view
    _function_space_views.push_back(V);
  }
}
//-----------------------------------------------------------------------------
