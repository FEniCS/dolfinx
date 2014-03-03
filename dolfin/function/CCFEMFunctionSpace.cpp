// Copyright (C) 2013-2014 Anders Logg
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
// Last changed: 2014-03-03

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include <dolfin/fem/CCFEMDofMap.h>
#include "FunctionSpace.h"
#include "CCFEMFunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CCFEMFunctionSpace::CCFEMFunctionSpace()
  : _multi_mesh(new MultiMesh()), _dofmap(new CCFEMDofMap())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CCFEMFunctionSpace::~CCFEMFunctionSpace()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t CCFEMFunctionSpace::dim() const
{
  dolfin_assert(_dofmap);
  return _dofmap->global_dimension();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const CCFEMDofMap> CCFEMFunctionSpace::dofmap() const
{
  dolfin_assert(_dofmap);
  return _dofmap;
}
//-----------------------------------------------------------------------------
std::size_t CCFEMFunctionSpace::num_parts() const
{
  return _function_spaces.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace>
CCFEMFunctionSpace::part(std::size_t i) const
{
  dolfin_assert(i < _function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
void
CCFEMFunctionSpace::add(std::shared_ptr<const FunctionSpace> function_space)
{
  _function_spaces.push_back(function_space);
  log(PROGRESS, "Added function space to CCFEM space; space has %d part(s).",
      _function_spaces.size());
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::add(const FunctionSpace& function_space)
{
  add(reference_to_no_delete_pointer(function_space));
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::build()
{
  begin(PROGRESS, "Building CCFEM function space.");

  // Build multi mesh
  _build_multi_mesh();

  // Build dofmap
  _build_dofmap();

  end();
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::_build_multi_mesh()
{
  // Clear multi mesh
  dolfin_assert(_multi_mesh);
  _multi_mesh->clear();

  // Add meshes
  for (std::size_t i = 0; i < num_parts(); i++)
    _multi_mesh->add(_function_spaces[i]->mesh());

  // Build multi mesh
  _multi_mesh->build();
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::_build_dofmap()
{
  begin(PROGRESS, "Building CCFEM dofmap.");

  // Clear dofmap
  dolfin_assert(_dofmap);
  _dofmap->clear();

  // Add dofmap for each part
  for (std::size_t i = 0; i < num_parts(); i++)
    _dofmap->add(_function_spaces[i]->dofmap());

  // Call function to build dofmap
  _dofmap->build(*this);

  end();
}
//-----------------------------------------------------------------------------
