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
// Last changed: 2014-04-24

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
const std::map<unsigned int, quadrature_rule> &
MultiMesh::quadrature_rule_cut_cells(std::size_t part) const

{
  dolfin_assert(part < num_parts());
  return _quadrature_rules_cut_cells[part];
}
//-----------------------------------------------------------------------------
quadrature_rule
MultiMesh::quadrature_rule_cut_cell(std::size_t part,
                                    unsigned int cell_index) const
{
  auto q = quadrature_rule_cut_cells(part);
  return q[cell_index];
}
//-----------------------------------------------------------------------------
const std::map<unsigned int, quadrature_rule>&
MultiMesh::quadrature_rule_overlap(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _quadrature_rules_overlap[part];
}
//-----------------------------------------------------------------------------
const std::map<unsigned int, std::vector<quadrature_rule> >&
MultiMesh::quadrature_rule_interface(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _quadrature_rules_interface[part];
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

  // FIXME: For collisions with meshes of same type we get three types
  // of quadrature rules: the cut cell qr, qr of the overlap part and
  // qr of the interface.

  // Build quadrature rules of the cut cells' overlap. Do this before
  // we build the quadrature rules of the cut cells
  _build_quadrature_rules_overlap();

  // Build quadrature rules of the cut cells
  _build_quadrature_rules_cut_cells();

  // Build quadrature rules of the interface in the cut cells
  //_build_quadrature_rules_interface();

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
  _collision_maps_cut_cells_boundary.clear();
  _quadrature_rules_cut_cells.clear();
  _quadrature_rules_overlap.clear();
  _quadrature_rules_interface.clear();
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

    // FIXME: what if the boundary mesh is empty?
    if (_boundary_meshes[i]->num_vertices()>0)
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
  _collision_maps_cut_cells_boundary.clear();

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
      const auto& boundary_collisions = _trees[i]->compute_collisions(*_boundary_trees[j]);

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
      const auto& domain_collisions = _trees[i]->compute_collisions(*_trees[j]);

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
void MultiMesh::_build_quadrature_rules_overlap()
{
  begin(PROGRESS, "Building quadrature rules of cut cells' overlap.");

  // FIXME: Make this a parameters
  const std::size_t order = 1;

  // Clear quadrature rules
  _quadrature_rules_overlap.clear();
  _quadrature_rules_interface.clear();

  // Resize quadrature rules
  _quadrature_rules_overlap.resize(num_parts());
  _quadrature_rules_interface.resize(num_parts());

  // FIXME: test prebuild map from boundary facets to full mesh cells
  // for all meshes: Loop over all boundary mesh facets to find the
  // full mesh cell which contains the facet. This is done in two
  // steps: Since the facet is on the boundary mesh, we first map this
  // facet to a facet in the full mesh using the
  // boundary_cell_map. Then we use the full_facet_cell_map to find
  // the corresponding cell in the full mesh. This cell is to match
  // the cutting_cell_no.

  // Build map from boundary facets to full mesh
  std::vector<std::vector<std::vector<std::size_t> > > full_to_bdry(num_parts());
  for (std::size_t part = 0; part < num_parts(); ++part)
  {
    full_to_bdry[part].resize(_meshes[part]->num_cells());

    // Get map from boundary mesh to facets of full mesh
    const std::size_t tdim_boundary
      = _boundary_meshes[part]->topology().dim();
    const auto& boundary_cell_map
      = _boundary_meshes[part]->entity_map(tdim_boundary);

    // Generate facet to cell connectivity for full mesh
    const std::size_t tdim = _meshes[part]->topology().dim();
    _meshes[part]->init(tdim_boundary, tdim);
    const MeshConnectivity& full_facet_cell_map
      = _meshes[part]->topology()(tdim_boundary, tdim);

    for (std::size_t boundary_facet = 0;
         boundary_facet < boundary_cell_map.size(); ++boundary_facet)
    {
      // Find the facet in the full mesh
      const std::size_t full_mesh_facet = boundary_cell_map[boundary_facet];

      // Find the cells in the full mesh (for interior facets we
      // can have 2 facets, but here we should only have 1)
      dolfin_assert(full_facet_cell_map.size(full_mesh_facet) == 1);
      const auto& full_cells = full_facet_cell_map(full_mesh_facet);
      full_to_bdry[part][full_cells[0]].push_back(boundary_facet);
    }
  }

  // Iterate over all parts
  for (std::size_t cut_part = 0; cut_part < num_parts(); cut_part++)
  {
    // Iterate over cut cells for current part
    const auto& cmap = collision_map_cut_cells(cut_part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      // Get cut cell
      const unsigned int cut_cell_index = it->first;
      const Cell cut_cell(*(_meshes[cut_part]), cut_cell_index);

      // Get dimensions
      const std::size_t tdim = cut_cell.mesh().topology().dim();
      const std::size_t gdim = cut_cell.mesh().geometry().dim();

      // Data structure for the volume triangulation of the cut_cell
      std::vector<double> volume_triangulation;

      // Data structure for the quadrature rule of this cell
      quadrature_rule volume_qr;

      // Data structure for the interface quadrature rule
      std::vector<quadrature_rule> interface_qr;

      // Data structure for the interface triangulation
      std::vector<double> interface_triangulation;

      // Iterate over cutting cells
      const auto& cutting_cells = it->second;
      for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
      {
        // Get cutting part and cutting cell
        const std::size_t cutting_part = jt->first;
        const std::size_t cutting_cell_index = jt->second;
        const Cell cutting_cell(*(_meshes[cutting_part]), cutting_cell_index);

        // Topology of this cut part
        const std::size_t tdim_boundary = _boundary_meshes[cutting_part]->topology().dim();

        // Must have the same topology at the moment (FIXME)
        dolfin_assert(cutting_cell.mesh().topology().dim() == tdim);

        // Data structure for local interface triangulation
        std::vector<double> local_interface_triangulation;

        // Data structure for the interface part quadrature rule
        quadrature_rule interface_part_qr;

        // Iterate over boundary cells
        for (auto boundary_cell_index : full_to_bdry[cutting_part][cutting_cell_index])
        {
          const Cell boundary_cell(*_boundary_meshes[cutting_part],
                                   boundary_cell_index);

          // Triangulate intersection of cut cell and boundary cell
          const auto triangulation_cut_boundary
            = cut_cell.triangulate_intersection(boundary_cell);

          // Add quadrature rule for triangulation
          if (triangulation_cut_boundary.size())
          {
            _add_quadrature_rule(interface_part_qr,
                                 triangulation_cut_boundary,
                                 tdim_boundary, gdim, order, 1);
          }

          // Triangulate intersection of boundary cell and previous volume triangulation
          const auto triangulation_boundary_prev_volume
            = IntersectionTriangulation::triangulate_intersection(boundary_cell,
                                                                  volume_triangulation,
                                                                  tdim);

          // Add quadrature rule for triangulation
          if (triangulation_boundary_prev_volume.size())
          {
            _add_quadrature_rule(interface_part_qr,
                                 triangulation_boundary_prev_volume,
                                 tdim_boundary, gdim, order, -1);
          }

          // Update triangulation
          local_interface_triangulation.insert(local_interface_triangulation.end(),
                                               triangulation_cut_boundary.begin(),
                                               triangulation_cut_boundary.end());
        }

        // Triangulate the intersection of the previous interface
        // triangulation and the cutting cell (to remove)
        const auto triangulation_prev_cutting
          = IntersectionTriangulation::triangulate_intersection(cutting_cell,
                                                                interface_triangulation,
                                                                tdim_boundary);

        // Add quadrature rule for triangulation
        if (triangulation_prev_cutting.size())
        {
          _add_quadrature_rule(interface_part_qr,
                               triangulation_prev_cutting,
                               tdim_boundary, gdim, order, -1);
        }

        // Update triangulation
        interface_triangulation.insert(interface_triangulation.end(),
                                       local_interface_triangulation.begin(),
                                       local_interface_triangulation.end());

        // Do the volume segmentation

        // Compute volume triangulation of intersection of cut and cutting cells
        const auto triangulation_cut_cutting
          = cut_cell.triangulate_intersection(cutting_cell);

        // Compute triangulation of intersection of cutting cell and
        // the (previous) volume triangulation
        const auto triangulation_cutting_prev
          = IntersectionTriangulation::triangulate_intersection(cutting_cell,
                                                                volume_triangulation,
                                                                tdim);

        // Add these new triangulations
        volume_triangulation.insert(volume_triangulation.end(),
                                    triangulation_cut_cutting.begin(),
                                    triangulation_cut_cutting.end());

        // Add quadrature rule with weights corresponding to the two
        // triangulations
        _add_quadrature_rule(volume_qr,
                             triangulation_cut_cutting,
                             tdim, gdim, order, 1);
        _add_quadrature_rule(volume_qr,
                             triangulation_cutting_prev,
                             tdim, gdim, order, -1);

        // Add quadrature rule for interface part
        interface_qr.push_back(interface_part_qr);
      }

      // Store quadrature rules for cut cell
      _quadrature_rules_overlap[cut_part][cut_cell_index] = volume_qr;
      _quadrature_rules_interface[cut_part][cut_cell_index] = interface_qr;
    }
  }

  end();
}
//-----------------------------------------------------------------------------
void MultiMesh::_build_quadrature_rules_cut_cells()
{
  begin(PROGRESS, "Building quadrature rules of cut cells.");

  // FIXME: Make this a parameter. Do we want to check to make sure we
  // have the same order in the overlapping part?
  const std::size_t order = 1;

  // Clear quadrature rules
  _quadrature_rules_cut_cells.clear();
  _quadrature_rules_cut_cells.resize(num_parts());

  // Iterate over all parts
  for (std::size_t cut_part = 0; cut_part < num_parts(); cut_part++)
  {
    // Iterate over cut cells for current part
    const auto& cmap = collision_map_cut_cells(cut_part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      // Get cut cell
      const unsigned int cut_cell_index = it->first;
      const Cell cut_cell(*(_meshes[cut_part]), cut_cell_index);

      // Get dimension
      const std::size_t gdim = cut_cell.mesh().geometry().dim();

      // Compute quadrature rule for the cell itself.
      auto qr = SimplexQuadrature::compute_quadrature_rule(cut_cell, order);

      // Get the quadrature rule for the overlapping part
      const auto& qr_overlap = _quadrature_rules_overlap[cut_part][cut_cell_index];

      // Add the quadrature rule for the overlapping part to the
      // quadrature rule of the cut cell with flipped sign
      _add_quadrature_rule(qr, qr_overlap, gdim, -1);

      // Store quadrature rule for cut cell
      _quadrature_rules_cut_cells[cut_part][cut_cell_index] = qr;
    }
  }

  end();
}
// //-----------------------------------------------------------------------------
// void MultiMesh::_build_quadrature_rules_interface()
// {
//   begin(PROGRESS, "Building quadrature rules of the interface in the cut cells.");

//   // FIXME: Make this a parameters
//   const std::size_t order = 1;

//   // Clear quadrature rules
//   _quadrature_rules_interface.clear();
//   _quadrature_rules_interface.resize(num_parts());

//   //double total_volume = 0;

//   // Iterate over all parts
//   for (std::size_t cut_part = 0; cut_part < num_parts(); cut_part++)
//   {
//     //std::cout << "% part " << cut_part << '\n';

//     // Iterate over cut cells for current part
//     const auto& cmap = collision_map_cut_cells(cut_part);
//     for (auto it = cmap.begin(); it != cmap.end(); ++it)
//     {
//       // Get cut cell
//       const unsigned int cut_cell_index = it->first;
//       const Cell cut_cell(*(_meshes[cut_part]), cut_cell_index);

//       //std::cout << "% cut cell " << cut_cell_index << '\n';

//       // Get dimensions
//       const std::size_t gdim = cut_cell.mesh().geometry().dim();

//       // Data structure for the quadrature rule of this cell
//       quadrature_rule quadrature_rule;

//       // Data structure for the total triangulation of the cut_cell
//       std::vector<double> total_triangulation;

//       // Iterate over cutting cells
//       const auto& cutting_cells = it->second;
//       for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
//       {
//         // Get cutting part and cutting cell
//         const std::size_t cutting_part = jt->first;
// 	const std::size_t cutting_cell_no = jt->second;
//         const Cell cutting_cell(*(_meshes[cutting_part]), cutting_cell_no);

// 	//std::cout << "% cutting part and cutting cell no " << cutting_part <<' '<<cutting_cell_no<<'\n';

// 	const std::size_t tdim = cutting_cell.mesh().topology().dim(); // FIXME?

// 	// Get map from boundary mesh to facets of full mesh
// 	const std::size_t tdim_boundary = _boundary_meshes[cutting_part]->topology().dim();
// 	const auto& boundary_cell_map = _boundary_meshes[cutting_part]->entity_map(tdim_boundary);

// 	// Generate facet -> cell connectivity
// 	_meshes[cutting_part]->init(tdim_boundary, tdim);
// 	const MeshConnectivity& full_facet_cell_map = _meshes[cutting_part]->topology()(tdim_boundary, tdim);

// 	for (std::size_t facet = 0; facet < boundary_cell_map.size(); ++facet)
// 	{
// 	  const std::size_t full_mesh_facet = boundary_cell_map[facet];
// 	  const auto size = full_facet_cell_map.size(full_mesh_facet);
// 	  dolfin_assert(size == 1);
// 	  const auto& full_cells = full_facet_cell_map(full_mesh_facet);

// 	  if (full_cells[0] == cutting_cell_no)
// 	  {
// 	    const Cell boundary_cell(*_boundary_meshes[cutting_part], facet);

// 	    // Triangulate the intersection of cut_cell and save
// 	    const auto triangulation = IntersectionTriangulation::triangulate_intersection(cut_cell, boundary_cell);
// 	    total_triangulation.insert(total_triangulation.end(),
// 				       triangulation.begin(),
// 				       triangulation.end());

// 	    // Create quadrature rules
// 	    _add_quadrature_rule(quadrature_rule,
// 				 triangulation,
// 				 tdim_boundary, gdim, order, 1);
// 	  }
// 	}
//       }

//       // Store quadrature rule for cut cell
//       _quadrature_rules_interface[cut_part][cut_cell_index] = quadrature_rule;
//     }
//   }

//   //std::cout << "\n\n"<<total_volume<<"\n\n";

//   end();
// }
//-----------------------------------------------------------------------------
void MultiMesh::_add_quadrature_rule(quadrature_rule& qr,
                                     const std::vector<double>& triangulation,
                                     std::size_t tdim,
                                     std::size_t gdim,
                                     std::size_t order,
                                     double factor) const
{
  // Iterate over simplices in triangulation
  const std::size_t offset = (tdim + 1)*gdim; // coordinates per simplex
  const std::size_t num_simplices = triangulation.size() / offset;
  for (std::size_t k = 0; k < num_simplices; k++)
  {
    // Get coordinates for current simplex in triangulation
    const double* x = &triangulation[0] + k*offset;

    // Compute quadrature rule for simplex
    const auto dqr = SimplexQuadrature::compute_quadrature_rule(x,
                                                               tdim,
                                                               gdim,
                                                               order);

    // Add quadrature rule
    _add_quadrature_rule(qr, dqr, gdim, factor);
  }
}
//-----------------------------------------------------------------------------
void MultiMesh::_add_quadrature_rule(quadrature_rule& qr,
                                     const quadrature_rule& dqr,
                                     std::size_t gdim,
                                     double factor) const
{
  // Get the number of points
  dolfin_assert(dqr.first.size() == gdim*dqr.second.size());
  const std::size_t num_points = dqr.second.size();

  // Append points and weights
  for (std::size_t i = 0; i < num_points; i++)
  {
    // Add point
    for (std::size_t j = 0; j < gdim; j++)
      qr.first.push_back(dqr.first[i*gdim + j]);

    // Add weight
    qr.second.push_back(factor*dqr.second[i]);
  }
}
//-----------------------------------------------------------------------------
