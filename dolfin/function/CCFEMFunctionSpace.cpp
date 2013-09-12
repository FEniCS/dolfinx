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
// Last changed: 2013-09-12

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include "FunctionSpace.h"
#include "CCFEMFunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CCFEMFunctionSpace::CCFEMFunctionSpace() : _dim(0)
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
  return _dim;
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::add(boost::shared_ptr<const FunctionSpace> function_space)
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

  // Get number of spaces
  const std::size_t num_spaces = _function_spaces.size();

  // Compute total dimension
  begin(PROGRESS, "Computing total dimension.");
  _dim = 0;
  for (std::size_t i = 0; i < num_spaces; i++)
  {
    const std::size_t d = _function_spaces[i]->dim();
    _dim += d;
    log(PROGRESS, "dim(V_%d) = %d", i, d);
  }
  end();
  log(PROGRESS, "Total dimension is %d.", _dim);

  // Build bounding box trees for all meshes
  begin(PROGRESS, "Building bounding box trees for all meshes.");
  _trees.clear();
  for (std::size_t i = 0; i < num_spaces; i++)
  {
    boost::shared_ptr<BoundingBoxTree> tree(new BoundingBoxTree());
    tree->build(*_function_spaces[i]->mesh());
    _trees.push_back(tree);
  }
  end();

  // Compute collisions between all meshes
  begin(PROGRESS, "Computing collisions between meshes.");
  for (std::size_t i = 0; i < num_spaces; i++)
  {
    for (std::size_t j = i + 1; j < num_spaces; j++)
    {
      log(PROGRESS, "Computing collisions for mesh %d overlapped by mesh %d.", i, j);
      _trees[i]->compute_collisions(*_trees[j]);
    }
  }
  end();

  end();
}
//-----------------------------------------------------------------------------
