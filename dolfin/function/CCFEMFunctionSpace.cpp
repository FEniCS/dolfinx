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
// Last changed: 2014-02-06

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/BoundaryMesh.h>
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

  // Extract meshes
  _meshes.clear();
  for (std::size_t i = 0; i < num_parts(); i++)
    _meshes.push_back(_function_spaces[i]->mesh());

  // Build dofmap
  _build_dofmap();

  // Build boundary meshes
  _build_boundary_meshes();

  // Build bounding box trees
  _build_bounding_box_trees();

  // Compute collision maps
  _build_collision_maps();

  end();
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::clear()
{
  dolfin_assert(_dofmap);

  _function_spaces.clear();
  _boundary_meshes.clear();
  _trees.clear();
  _boundary_trees.clear();
  _dofmap->clear();
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
void CCFEMFunctionSpace::_build_boundary_meshes()
{
  begin(PROGRESS, "Building boundary meshes.");

  // Clear boundary meshes
  _boundary_meshes.clear();

  // Build boundary mesh for each part
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    boost::shared_ptr<BoundaryMesh>
      boundary_mesh(new BoundaryMesh(*_meshes[i], "exterior"));
    _boundary_meshes.push_back(boundary_mesh);
  }

  end();
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::_build_bounding_box_trees()
{
  begin(PROGRESS, "Building bounding box trees for all meshes.");

  // Clear bounding box trees
  _trees.clear();
  _boundary_trees.clear();

  // Build trees for each part
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    // Build tree for mesh
    boost::shared_ptr<BoundingBoxTree> tree(new BoundingBoxTree());
    tree->build(*_meshes[i]);
    _trees.push_back(tree);

    // Build tree for boundary mesh
    boost::shared_ptr<BoundingBoxTree> boundary_tree(new BoundingBoxTree());
    boundary_tree->build(*_boundary_meshes[i]);
    _boundary_trees.push_back(boundary_tree);
  }

  end();
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::_build_collision_maps()
{
  begin(PROGRESS, "Building collision maps.");

  // Iterate over all parts
  for (std::size_t i = 0; i < num_parts(); i++)
  {


    // Iterate over covering parts (with higher part number)
    for (std::size_t j = i + 1; j < num_parts(); j++)
    {
      log(PROGRESS, "Computing collisions for mesh %d overlapped by mesh %d.", i, j);

      log(PROGRESS, "  Computing domain collisions.");
      auto domain_collisions = _trees[i]->compute_collisions(*_trees[j]);
      log(PROGRESS, "  Computing boundary collisions.");
      auto boundary_collisions = _trees[i]->compute_collisions(*_boundary_trees[j]);
    }
  }

  end();
}
//-----------------------------------------------------------------------------
