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
// Modified by August Johansson 2014
//
// First added:  2013-08-05
// Last changed: 2014-03-13

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include "MultiMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMesh::MultiMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MultiMesh::~MultiMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t MultiMesh::num_parts() const
{
  return _meshes.size();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Mesh> MultiMesh::part(std::size_t i) const
{
  dolfin_assert(i < _meshes.size());
  return _meshes[i];
}
//-----------------------------------------------------------------------------
const std::vector<unsigned int>&
MultiMesh::uncut_cells(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _uncut_cells[part];
}
//-----------------------------------------------------------------------------
const std::vector<unsigned int>&
MultiMesh::cut_cells(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _cut_cells[part];
}
//-----------------------------------------------------------------------------
const std::vector<unsigned int>&
MultiMesh::covered_cells(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _covered_cells[part];
}
//-----------------------------------------------------------------------------
const std::map<unsigned int,
               std::vector<std::pair<std::size_t, unsigned int> > >&
MultiMesh::collision_map_cut_cells(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _collision_maps_cut_cells[part];
}
//-----------------------------------------------------------------------------
const std::map<unsigned int,
               std::pair<std::vector<double>, std::vector<double> > > &
MultiMesh::quadrature_rule_cut_cells(std::size_t part) const

{
  dolfin_assert(part < num_parts());
  return _quadrature_rules_cut_cells[part];
}
//-----------------------------------------------------------------------------
void MultiMesh::add(std::shared_ptr<const Mesh> mesh)
{
  _meshes.push_back(mesh);
  log(PROGRESS, "Added mesh to multimesh; multimesh has %d part(s).",
      _meshes.size());
}
//-----------------------------------------------------------------------------
void MultiMesh::add(const Mesh& mesh)
{
  add(reference_to_no_delete_pointer(mesh));
}
//-----------------------------------------------------------------------------
void MultiMesh::build()
{
  begin(PROGRESS, "Building multimesh.");

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
void MultiMesh::clear()
{
  _boundary_meshes.clear();
  _trees.clear();
  _boundary_trees.clear();
  _uncut_cells.clear();
  _cut_cells.clear();
  _covered_cells.clear();
  _collision_maps_cut_cells.clear();
  _quadrature_rules_cut_cells.clear();
}
//-----------------------------------------------------------------------------
void MultiMesh::_build_boundary_meshes()
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
void MultiMesh::_build_bounding_box_trees()
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
void MultiMesh::_build_collision_maps()
{
  begin(PROGRESS, "Building collision maps.");

  // Clear collision maps
  _uncut_cells.clear();
  _cut_cells.clear();
  _covered_cells.clear();
  _collision_maps_cut_cells.clear();

  // Iterate over all parts
  for (std::size_t i = 0; i < num_parts(); i++)
  {
    // Extract uncut, cut and covered cells:
    //
    // 0: uncut   = cell not colliding with any higher domain
    // 1: cut     = cell colliding with some higher boundary and is not covered
    // 2: covered = cell colliding with some higher domain but not its boundary

    // Create vector of markers for cells in part `i` (0, 1, or 2)
    std::vector<char> markers(_meshes[i]->num_cells(), 0);

    // Create local array for marking boundary collisions for cells in
    // part `i`. Note that in contrast to the markers above which are
    // global to part `i`, these markers are local to the collision
    // between part `i` and part `j`.
    std::vector<bool> collides_with_boundary(_meshes[i]->num_cells());

    // Create empty collision map for cut cells in part `i`
    std::map<unsigned int, std::vector<std::pair<std::size_t, unsigned int> > >
      collision_map_cut_cells;

    // Iterate over covering parts (with higher part number)
    for (std::size_t j = i + 1; j < num_parts(); j++)
    {
      log(PROGRESS, "Computing collisions for mesh %d overlapped by mesh %d.", i, j);

      // Reset boundary collision markers
      std::fill(collides_with_boundary.begin(), collides_with_boundary.end(), false);

      // Compute domain-boundary collisions
      auto boundary_collisions = _trees[i]->compute_collisions(*_boundary_trees[j]);

      // Iterate over boundary collisions
      for (auto it = boundary_collisions.first.begin();
           it != boundary_collisions.first.end(); ++it)
      {
        // Mark that cell collides with boundary
        collides_with_boundary[*it] = true;

        // Mark as cut cell if not previously covered
        if (markers[*it] != 2)
        {
          // Mark as cut cell
          markers[*it] = 1;

          // Add empty list of collisions into map if it does not exist
          if (collision_map_cut_cells.find(*it) == collision_map_cut_cells.end())
          {
            std::vector<std::pair<std::size_t, unsigned int> > collisions;
            collision_map_cut_cells[*it] = collisions;
          }
        }
      }

      // Compute domain-domain collisions
      auto domain_collisions = _trees[i]->compute_collisions(*_trees[j]);

      // Iterate over domain collisions
      dolfin_assert(domain_collisions.first.size() == domain_collisions.second.size());
      for (std::size_t k = 0; k < domain_collisions.first.size(); k++)
      {
        // Get the two colliding cells
        auto cell_i = domain_collisions.first[k];
        auto cell_j = domain_collisions.second[k];

        // Store collision in collision map if we have a cut cell
        if (markers[cell_i] == 1)
        {
          auto it = collision_map_cut_cells.find(cell_i);
          dolfin_assert(it != collision_map_cut_cells.end());
          it->second.push_back(std::make_pair(j, cell_j));
        }

        // Mark cell as covered if it does not collide with boundary
        if (!collides_with_boundary[cell_i])
        {
          // Remove from collision map if previously marked as as cut cell
          if (markers[cell_i] == 1)
          {
            dolfin_assert(collision_map_cut_cells.find(cell_i) != collision_map_cut_cells.end());
            collision_map_cut_cells.erase(cell_i);
          }

          // Mark as covered cell (may already be marked)
          markers[cell_i] = 2;
        }
      }
    }

    // Extract uncut, cut and covered cells from markers
    std::vector<unsigned int> uncut_cells;
    std::vector<unsigned int> cut_cells;
    std::vector<unsigned int> covered_cells;
    for (unsigned int c = 0; c < _meshes[i]->num_cells(); c++)
    {
      switch (markers[c])
      {
      case 0:
        uncut_cells.push_back(c);
        break;
      case 1:
        cut_cells.push_back(c);
        break;
      default:
        covered_cells.push_back(c);
      }
    }

    // Store data for this mesh
    _uncut_cells.push_back(uncut_cells);
    _cut_cells.push_back(cut_cells);
    _covered_cells.push_back(covered_cells);
    _collision_maps_cut_cells.push_back(collision_map_cut_cells);

    // Report results
    log(PROGRESS, "Part %d has %d uncut cells, %d cut cells, and %d covered cells.",
        i, uncut_cells.size(), cut_cells.size(), covered_cells.size());
  }

  end();
}
//-----------------------------------------------------------------------------
void MultiMesh::_build_quadrature_rules()
{
  begin(PROGRESS, "Building quadrature rules.");

  // FIXME: Make this a parameters
  const std::size_t order = 1;

  // Clear quadrature rules
  _quadrature_rules_cut_cells.clear();
  _quadrature_rules_cut_cells.resize(num_parts());

  // Iterate over all parts
  for (std::size_t cut_part = 0; cut_part < num_parts(); cut_part++)
  {
    // Iterate over cut cells for current part
    const auto cmap = collision_map_cut_cells(cut_part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      // Get cut cell
      const unsigned int cut_cell_index = it->first;
      const Cell cut_cell(*(_meshes[cut_part]), cut_cell_index);

      // Compute quadrature rule for the cell itself
      auto quadrature_rule = SimplexQuadrature::compute_quadrature_rule(cut_cell, order);

      // Iterate over cutting cells
      auto cutting_cells = it->second;
      for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
      {
        // Get cutting part and cutting cell
        const std::size_t cutting_part = jt->first;
        const Cell cutting_cell(*(_meshes[cutting_part]), jt->second);

        // Subtract quadrature rule for intersection
        _add_quadrature_rule(quadrature_rule, cut_cell, cutting_cell, order, -1);
      }

      // Store quadrature rule for cut cell
      _quadrature_rules_cut_cells[cut_part][cut_cell_index] = quadrature_rule;
    }
  }

  end();
}
//-----------------------------------------------------------------------------
void
MultiMesh::_add_quadrature_rule(std::pair<std::vector<double>,
                                          std::vector<double> > quadrature_rule,
                                const Cell& cell_0,
                                const Cell& cell_1,
                                std::size_t order,
                                double factor) const
{
  // Get dimensions
  const std::size_t tdim = cell_0.mesh().topology().dim();
  const std::size_t gdim = cell_1.mesh().geometry().dim();

  // Compute triangulation of intersection between cells
  auto triangulation = cell_0.triangulate_intersection(cell_1);

  // Iterate over simplices in triangulation
  const std::size_t offset = (tdim + 1)*gdim; // coordinates per simplex
  const std::size_t num_intersections = triangulation.size() / offset;
  for (std::size_t k = 0; k < num_intersections; k++)
  {
    // Get coordinates for current simplex in triangulation
    const double* x = &triangulation[0] + k*offset;

    // Compute quadrature rule for simplex
    auto q = SimplexQuadrature::compute_quadrature_rule(x, tdim, gdim, order);

    // Add quadrature rule
    dolfin_assert(gdim*q.first.size() == q.second.size());
    const std::size_t num_points = q.first.size();
    for (std::size_t i = 0; i < num_points; i++)
    {
      quadrature_rule.first.push_back(factor*q.first[i]);
      for (std::size_t j = 0; j < gdim; j++)
        quadrature_rule.second.push_back(q.second[i*gdim + j]);
    }
  }
}
//-----------------------------------------------------------------------------
