// Copyright (C) 2013 Anders Logg
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
// Last changed: 2013-09-19

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/fem/CCFEMDofMap.h>
#include "FunctionSpace.h"
#include "CCFEMFunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CCFEMFunctionSpace::CCFEMFunctionSpace() : _dofmap(new CCFEMDofMap())
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
boost::shared_ptr<const CCFEMDofMap> CCFEMFunctionSpace::dofmap() const
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
boost::shared_ptr<const FunctionSpace>
CCFEMFunctionSpace::part(std::size_t i) const
{
  dolfin_assert(i < _function_spaces.size());
  return _function_spaces[i];
}
//-----------------------------------------------------------------------------
void
CCFEMFunctionSpace::add(boost::shared_ptr<const FunctionSpace> function_space)
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

  // Build dofmap
  dolfin_assert(_dofmap);
  _dofmap->clear();
  for (std::size_t i = 0; i < num_parts(); i++)
    _dofmap->add(_function_spaces[i]->dofmap());
  _dofmap->build(*this);

  // Build bounding box trees for all meshes
  begin(PROGRESS, "Building bounding box trees for all meshes.");
  _trees.clear();
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    boost::shared_ptr<BoundingBoxTree> tree(new BoundingBoxTree());
    tree->build(*_function_spaces[i]->mesh());
    _trees.push_back(tree);
  }
  end();

  // Compute collisions between all meshes
  begin(PROGRESS, "Computing collisions between meshes.");
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    for (std::size_t j = i + 1; j < num_parts(); j++)
    {
      log(PROGRESS, "Computing collisions for mesh %d overlapped by mesh %d.", i, j);
      _trees[i]->compute_collisions(*_trees[j]);
    }
  }
  end();

  end();
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::clear()
{
  dolfin_assert(_dofmap);

  _function_spaces.clear();
  _trees.clear();
  _dofmap->clear();
}
//-----------------------------------------------------------------------------
