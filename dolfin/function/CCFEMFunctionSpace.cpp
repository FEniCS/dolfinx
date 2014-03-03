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
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/SimplexQuadrature.h>
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
const std::vector<unsigned int>&
CCFEMFunctionSpace::uncut_cells(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _uncut_cells[part];
}
//-----------------------------------------------------------------------------
const std::vector<unsigned int>&
CCFEMFunctionSpace::cut_cells(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _cut_cells[part];
}
//-----------------------------------------------------------------------------
const std::vector<unsigned int>&
CCFEMFunctionSpace::covered_cells(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _covered_cells[part];
}
//-----------------------------------------------------------------------------
const std::map<unsigned int,
               std::vector<std::pair<std::size_t, unsigned int> > >&
  CCFEMFunctionSpace::collision_map_cut_cells(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _collision_map_cut_cells[part];
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

  // Build collision maps
  _build_collision_maps();

  // Build quadrature rules
  _build_quadrature_rules();

  end();
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
    std::shared_ptr<BoundaryMesh>
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
    std::shared_ptr<BoundingBoxTree> tree(new BoundingBoxTree());
    tree->build(*_meshes[i]);
    _trees.push_back(tree);

    // Build tree for boundary mesh
    std::shared_ptr<BoundingBoxTree> boundary_tree(new BoundingBoxTree());
    boundary_tree->build(*_boundary_meshes[i]);
    _boundary_trees.push_back(boundary_tree);
  }

  end();
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::_build_collision_maps()
{
  begin(PROGRESS, "Building collision maps.");

  // Clear collision maps
  _uncut_cells.clear();
  _cut_cells.clear();
  _covered_cells.clear();
  _collision_map_cut_cells.clear();

  // Iterate over all parts
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    // Collision map for cut cells in mesh `i`
    std::map<unsigned int, std::vector<std::pair<std::size_t, unsigned int> > >
      collision_map_cut_cells;

    // Markers for collisions with domain
    std::vector<bool> markers_domain(_meshes[i]->num_cells());
    std::fill(markers_domain.begin(), markers_domain.end(), false);

    // Iterate over covering parts (with higher part number)
    for (std::size_t j = i + 1; j < num_parts(); j++)
    {
      log(PROGRESS, "Computing collisions for mesh %d overlapped by mesh %d.", i, j);

      // Compute boundary collisions
      auto boundary_collisions = _trees[i]->compute_collisions(*_boundary_trees[j]);

      // Iterate over boundary collisions
      for (auto it = boundary_collisions.first.begin();
           it != boundary_collisions.first.end(); ++it)
      {
        // Add empty list of collisions into map if it does not exist
        if (collision_map_cut_cells.find(*it) == collision_map_cut_cells.end())
        {
          std::vector<std::pair<std::size_t, unsigned int> > collisions;
          collision_map_cut_cells[*it] = collisions;
        }
      }

      // Compute domain collisions
      auto domain_collisions = _trees[i]->compute_collisions(*_trees[j]);

      // Iterate over domain collisions
      dolfin_assert(domain_collisions.first.size() == domain_collisions.second.size());
      for (std::size_t k = 0; k < domain_collisions.first.size(); k++)
      {
        // Get the two colliding cells
        auto cell_i = domain_collisions.first[k];
        auto cell_j = domain_collisions.second[k];

        // Mark collision for first cell
        markers_domain[cell_i] = true;

        // Add to collision map if we find a cut cell
        auto it = collision_map_cut_cells.find(cell_i);
        if (it != collision_map_cut_cells.end())
          it->second.push_back(std::make_pair(j, cell_j));
      }
    }

    // Extract uncut, cut and covered cells:
    //
    // uncut   = cell not colliding with any other domain
    // cut     = cell colliding with some other boundary
    // covered = cell colliding with some other domain but no boundary

    // Iterate over cells and check markers
    std::vector<unsigned int> uncut_cells;
    std::vector<unsigned int> cut_cells;
    std::vector<unsigned int> covered_cells;
    for (unsigned int c = 0; c < _meshes[i]->num_cells(); c++)
    {
      if (!markers_domain[c])
        uncut_cells.push_back(c);
      else if (collision_map_cut_cells.find(c) != collision_map_cut_cells.end())
        cut_cells.push_back(c);
      else
        covered_cells.push_back(c);
    }

    // Store data for this mesh
    _uncut_cells.push_back(uncut_cells);
    _cut_cells.push_back(cut_cells);
    _covered_cells.push_back(covered_cells);
    _collision_map_cut_cells.push_back(collision_map_cut_cells);

    // Report results
    log(PROGRESS, "Part %d has %d uncut cells, %d cut cells, and %d covered cells.",
        i, uncut_cells.size(), cut_cells.size(), covered_cells.size());
  }

  end();
}
//-----------------------------------------------------------------------------
void CCFEMFunctionSpace::_build_quadrature_rules()
{
  begin(PROGRESS, "Building quadrature rules.");

  // FIXME: Make this a parameters
  const std::size_t order = 1;

  // Clear quadrature rules
  _quadrature_rules_cut_cells.clear();

  // Iterate over all parts
  for (std::size_t cut_part = 0; cut_part < num_parts(); cut_part++)
  {
    // Iterate over cut cells for current part
    const auto cmap = collision_map_cut_cells(cut_part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      // Get cut cell
      const Cell cut_cell(*(_meshes[cut_part]), it->first);

      // Get dimensions
      const std::size_t tdim = cut_cell.mesh().topology().dim();
      const std::size_t gdim = cut_cell.mesh().geometry().dim();

      // Iterate over cutting cells
      auto cutting_cells = it->second;
      for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
      {
        // Get cutting part and cutting cell
        const std::size_t cutting_part = jt->first;
        const Cell cutting_cell(*(_meshes[cutting_part]), jt->second);

        // Compute triangulation of intersection between cut and cutting cell
        auto triangulation = cut_cell.triangulate_intersection(cutting_cell);

        // Iterate over simplices in triangulation
        const std::size_t offset = (tdim + 1)*gdim; // coordinates per simplex
        const std::size_t num_intersections = triangulation.size() / offset;
        for (std::size_t k = 0; k < num_intersections; k++)
        {
          // Get coordinates for current simplex in triangulation
          const double* coordinates = &triangulation[0] + k*offset;

          // Compute quadrature rule for simplex
          auto quadrature_rule
            = SimplexQuadrature::compute_quadrature_rule(coordinates, tdim, gdim, order);
        }
      }
    }
  }

  end();
}
//-----------------------------------------------------------------------------
