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
// Modified by August Johansson 2015
//
// First added:  2013-08-05
// Last changed: 2015-06-03


#include <dolfin/log/log.h>
#include <dolfin/plot/plot.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include "Cell.h"
#include "Facet.h"
#include "BoundaryMesh.h"
#include "MeshFunction.h"
#include "MultiMesh.h"
// FIXME August
#include <dolfin/dolfin_simplex_tools.h>
#define Augustdebug

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMesh::MultiMesh()
{
  // Set parameters
  parameters = default_parameters();
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
               std::vector<std::pair<std::size_t, unsigned int>>>&
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
const std::map<unsigned int, std::vector<quadrature_rule>>&
  MultiMesh::quadrature_rule_overlap(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _quadrature_rules_overlap[part];
}
//-----------------------------------------------------------------------------
const std::map<unsigned int, std::vector<quadrature_rule>>&
  MultiMesh::quadrature_rule_interface(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _quadrature_rules_interface[part];
}
//-----------------------------------------------------------------------------
const std::map<unsigned int, std::vector<std::vector<double>>>&
  MultiMesh::facet_normals(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _facet_normals[part];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const BoundingBoxTree>
MultiMesh::bounding_box_tree(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _trees[part];
}
//-----------------------------------------------------------------------------
std::shared_ptr<const BoundingBoxTree>
MultiMesh::bounding_box_tree_boundary(std::size_t part) const
{
  dolfin_assert(part < num_parts());
  return _boundary_trees[part];
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
    std::map<unsigned int, std::vector<std::pair<std::size_t, unsigned int>>>
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
            std::vector<std::pair<std::size_t, unsigned int>> collisions;
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

  // Get quadrature order
  const std::size_t quadrature_order = parameters["quadrature_order"];

  // Clear quadrature rules
  _quadrature_rules_overlap.clear();
  _quadrature_rules_interface.clear();

  // Resize quadrature rules
  _quadrature_rules_overlap.resize(num_parts());
  _quadrature_rules_interface.resize(num_parts());

  // Clear and resize facet normals
  _facet_normals.clear();
  _facet_normals.resize(num_parts());

  // FIXME: test prebuild map from boundary facets to full mesh cells
  // for all meshes: Loop over all boundary mesh facets to find the
  // full mesh cell which contains the facet. This is done in two
  // steps: Since the facet is on the boundary mesh, we first map this
  // facet to a facet in the full mesh using the
  // boundary_cell_map. Then we use the full_facet_cell_map to find
  // the corresponding cell in the full mesh. This cell is to match
  // the cutting_cell_no.

  // Build map from boundary facets to full mesh
  std::vector<std::vector<std::vector<std::pair<std::size_t, std::size_t>>>>
    full_to_bdry(num_parts());
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
      full_to_bdry[part][full_cells[0]].push_back(std::make_pair(boundary_facet,
                                                                 full_mesh_facet));
    }
  }

  std::cout.precision(15);

  for (std::size_t cut_part = 0; cut_part < num_parts(); cut_part++)
  {
    // Iterate over cut cells for current part
    const auto& cmap = collision_map_cut_cells(cut_part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      // Get cut cell
      const unsigned int cut_cell_index = it->first;
      const Cell cut_cell(*(_meshes[cut_part]), cut_cell_index);
      std::cout << tools::drawtriangle(cut_cell);

      // Loop over all cutting cells to construct the polyhedra to be
      // used in the inclusion-exclusion principle
      for (auto jt = it->second.begin(); jt != it->second.end(); jt++)
      {
	// Get cutting part and cutting cell
        const std::size_t cutting_part = jt->first;
        const std::size_t cutting_cell_index = jt->second;
        const Cell cutting_cell(*(_meshes[cutting_part]), cutting_cell_index);
	std::cout << tools::drawtriangle(cutting_cell);
      }
    }
  }
  PPause;



  // Iterate over all parts
  for (std::size_t cut_part = 0; cut_part < num_parts(); cut_part++)
  {
#ifdef Augustdebug
    std::cout << "----- cut part: " << cut_part <<std::endl;
    tools::dolfin_write_medit_triangles("cut_part",*(_meshes[cut_part]),cut_part);
    //double areapos = 0, areaminus = 0;
#endif

    // Iterate over cut cells for current part
    const auto& cmap = collision_map_cut_cells(cut_part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
#ifdef Augustdebug
      std::cout << "-------- new cut cell\n";
#endif

      // Get cut cell
      const unsigned int cut_cell_index = it->first;
      const Cell cut_cell(*(_meshes[cut_part]), cut_cell_index);

      // Get dimensions
      const std::size_t tdim = cut_cell.mesh().topology().dim();
      const std::size_t gdim = cut_cell.mesh().geometry().dim();

      // Data structure for the overlap quadrature rule
      std::vector<quadrature_rule> overlap_qr;

      // Data structure for the first intersections (this is the first
      // stage in the inclusion exclusion principle). These are the
      // polyhedra to be used in the exlusion inclusion.
      std::vector<std::pair<std::size_t, Polyhedron> > initial_polyhedra;

      // Loop over all cutting cells to construct the polyhedra to be
      // used in the inclusion-exclusion principle
      for (auto jt = it->second.begin(); jt != it->second.end(); jt++)
      {
	  // Get cutting part and cutting cell
        const std::size_t cutting_part = jt->first;
        const std::size_t cutting_cell_index = jt->second;
        const Cell cutting_cell(*(_meshes[cutting_part]), cutting_cell_index);

#ifdef Augustdebug
	{
	  std::cout << "\ncut cutting (cutting part=" << cutting_part << ")" << std::endl;
	  std::cout << tools::drawtriangle(cut_cell,"'y'")<<tools::drawtriangle(cutting_cell,"'m'")<<std::endl;
	}
#endif

  	// Only allow same type of cell for now
      	dolfin_assert(cutting_cell.mesh().topology().dim() == tdim);
      	dolfin_assert(cutting_cell.mesh().geometry().dim() == gdim);

  	// Compute the intersection (a polyhedron)
  	const std::vector<double> intersection
	  = IntersectionTriangulation::triangulate_intersection(cut_cell,
								cutting_cell);
	const Polyhedron polyhedron = convert(intersection, tdim, gdim);

#ifdef Augustdebug
	{
	  std::cout << "intersection (size="<<intersection.size()<<": ";
	  for (std::size_t i = 0; i < intersection.size(); ++i)
	    std::cout << intersection[i]<<' ';
	  std::cout<<")\n";
	  if (polyhedron.size())
	  {
	    for (const auto simplex: polyhedron)
	      std::cout << tools::drawtriangle(simplex,"'k'");
	    std::cout << std::endl;
	    std::cout << "areas=[";
	    for (const auto simplex: polyhedron)
	      std::cout << tools::area(simplex)<<' ';
	    std::cout << "];"<<std::endl;
	  }
	}
#endif

	// Flip triangles in polyhedron to maximize minimum angle
	//const bool flipped = false;//maximize_minimum_angle(polyhedron);


	// Test only include large polyhedra
	double area = 0;
	for (const auto simplex: polyhedron)
	  area += std::abs(tools::area(simplex));
	if (std::isfinite(area) and area > DOLFIN_EPS_LARGE)
	{
	  // Store key and polyhedron
	  initial_polyhedra.push_back(std::make_pair(initial_polyhedra.size(),
						     polyhedron));
	}
      }
      //PPause;

      // Exclusion-inclusion principle. There are N stages in the
      // principle, where N = polyhedra.size(). The first stage is
      // simply the polyhedra themselves A, B, C, ... etc. The second
      // stage is for the pairwise intersections A \cap B, A \cap C, B
      // \cap C, etc, with different sign. There are
      // n_choose_k(N,stage) intersections for each stage.

      // Data structure for storing the previous intersections: the key
      // and the intersections.
      const std::size_t N = initial_polyhedra.size();
      std::vector<std::pair<std::vector<std::size_t>,
			    Polyhedron> > previous_intersections(N);
      for (std::size_t i = 0; i < N; ++i)
	previous_intersections[i]
	  = std::make_pair(std::vector<std::size_t>(1, initial_polyhedra[i].first),
			   initial_polyhedra[i].second);

      // Do stage = 1 up to stage = polyhedra.size in the
      // principle. Recall that stage 1 is the pairwise
      // intersections. There are up to n_choose_k(N,stage)
      // intersections in each stage (there may be less). The
      // intersections are found using the polyhedra data and the
      // previous_intersections data. We only have to intersect if the key doesn't
      // contain the polyhedron.

      // std::cout << std::endl << "initial setup done, resulted in " << N << " polyhedra to be used in the inclusion exclusion\n\n";

      // // The big data structure
      // std::vector<std::vector<std::pair<std::vector<std::size_t>, Polyhedron> > > all_intersections(N);

      // // Maybe not needed:
      // all_intersections[0] = previous_intersections; //initial_polyhedra;

      // Add quadrature rule for stage 0 (always positive)
      {
	quadrature_rule overlap_part_qr;
	const std::size_t sign = 1;
	for (const auto polyhedron: previous_intersections)
	  for (const auto simplex: polyhedron.second)
	  {
	    //areapos += tools::area(simplex);
	    std::vector<double> x = convert(simplex, tdim, gdim);
	    _add_quadrature_rule(overlap_part_qr, x,
				 tdim, gdim, quadrature_order, sign);
	  }

	// Add quadrature rule for overlap part
	overlap_qr.push_back(overlap_part_qr);
      }




      for (std::size_t stage = 1; stage < N; ++stage)
      {
	// std::cout << "----------------- stage " << stage << std::endl;

      	// Structure for storing new intersections
      	std::vector<std::pair<std::vector<std::size_t>,
			      Polyhedron> > new_intersections;

      	// Loop over all intersections from the previous stage
      	for (const auto previous_polyhedron: previous_intersections)
      	{
      	  // Loop over all initial polyhedra.
      	  for (const auto initial_polyhedron: initial_polyhedra)
      	  {
	    // std::cout << "check keys from previous_polyhedron: ";
	    // for (const auto key: previous_polyhedron.first)
	    //   std::cout << key <<' ';
	    // std::cout << '\n';
	    // std::cout << "initial key: " << initial_polyhedron.first << std::endl;

	    // test: only check if initial_polyhedron key <
	    // previous_polyhedron key[0]
	    if (initial_polyhedron.first < previous_polyhedron.first[0])
	    {

	      // {
	      // 	for (const auto previous_simplex: previous_polyhedron.second)
	      // 	  std::cout << tools::drawtriangle(previous_simplex,"'b'");
	      // 	std::cout << '\n';
	      // 	for (const auto initial_simplex: initial_polyhedron.second)
	      // 	  std::cout << tools::drawtriangle(initial_simplex,"'r'");
	      // 	std::cout<<'\n';
	      // }


	      // We want to save the intersection of the previous
	      // polyhedron and the initial polyhedron in one single
	      // polyhedron.
	      Polyhedron new_polyhedron;
	      std::vector<std::size_t> new_keys;

	      // Loop over all simplices in the initial_polyhedron and
	      // the previous_polyhedron and append the intersection of
	      // these to the new_polyhedron
	      bool any_intersections = false;

	      for (const auto previous_simplex: previous_polyhedron.second)
	      {
		for (const auto initial_simplex: initial_polyhedron.second)
		{

		  // Compute the intersection (a polyhedron)
		  const std::vector<double> ii
		    = IntersectionTriangulation::
		    triangulate_intersection(initial_simplex, tdim,
					     previous_simplex, tdim, gdim);

		  if (ii.size())
		  {
		    //any_intersections = true;

		    // To save all intersections as a single
		    // polyhedron, we don't call this a polyhedron
		    // yet, but rather a std::vector<Simplex> since we
		    // are still filling the polyhedron with simplices
		    std::vector<Simplex> pii = convert(ii, tdim, gdim);
		    // Test only add if area is large
		    for (const auto simplex: pii)
		    {
		      const double area = tools::area(simplex);
		      if (std::isfinite(area) and area > DOLFIN_EPS_LARGE)
		      //if (std::isfinite(area) and area > 1e-13)
		      {
			//std::cout << "added simplex with area " << area << std::endl;

			new_polyhedron.push_back(simplex);
			any_intersections = true;
		      }
		      // else
		      // {
		      // 	std::cout << "skipped simplex with area = "  << area << std::endl;

		      // 	// // debug
		      // 	// if (!std::isfinite(area))
		      // 	// {
		      // 	//   for (const auto pt: simplex)
		      // 	//     for (int d = 0; d < 2; ++d)
		      // 	//       std::cout<<std::setprecision(16) << pt[d]<<' ';
		      // 	//   std::cout << std::endl;
		      // 	// }
		      // }
		    }

#ifdef Augustdebug
		    {
		      std::cout << '\n'<<tools::drawtriangle(previous_simplex,"'b'")
		    		<< tools::drawtriangle(initial_simplex,"'r'")<<std::endl;
		      std::cout << "areas: " << tools::area(previous_simplex)<<' '<<tools::area(initial_simplex)<<'\n';
		      const double min_area = std::min(tools::area(previous_simplex), tools::area(initial_simplex));

		      std::cout << "resulting intersection:\n";
		      for (const auto simplex: pii)
		    	std::cout << tools::drawtriangle(simplex,"'g'");
		      std::cout<<'\n';
		      double intersection_area = 0;
		      std::cout << "areas: ";
		      for (const auto simplex: pii) {
		    	intersection_area += tools::area(simplex);
		    	std::cout << tools::area(simplex) <<' ';
		      }
		      std::cout<<'\n';
		      if (intersection_area >= min_area) { std::cout << "Warning, intersection area ~ minimum area\n"; /*PPause;*/ }
		    }
#endif
		  }
		}
	      }

	      if (any_intersections)
	      {
		new_keys.push_back(initial_polyhedron.first);
		new_keys.insert(new_keys.end(),
				previous_polyhedron.first.begin(),
				previous_polyhedron.first.end());
		// std::cout << "new keys: ";
		// for (const auto key: new_keys)
		//   std::cout << key <<' ';
		// std::cout<<std::endl;


		// Test improve quality
		//maximize_minimum_angle(new_polyhedron);


		// Save data
		new_intersections.push_back(std::make_pair(new_keys, new_polyhedron));
		//PPause;
	      }

	    }
	  }
      	}


	// {
	//   std::cout << "\n summarize at stage="<<stage<<" and part=" << cut_part<< '\n';
	//   std::cout << "the previous intersections were:\n";
	//   for (const auto previous_polyhedron: previous_intersections)
	//   {
	//     for (const auto key: previous_polyhedron.first)
	//       std::cout << key<<' ';
	//     std::cout << "   ";
	//   }
	//   std::cout << '\n';
	//   for (const auto previous_polyhedron: previous_intersections)
	//   {
	//     for (const auto simplex: previous_polyhedron.second)
	//       std::cout << tools::drawtriangle(simplex);
	//     std::cout << "    ";
	//   }
	//   std::cout << '\n';


	//   std::cout << "the new intersections are:\n";
	//   for (const auto new_polyhedron: new_intersections)
	//   {
	//     for (const auto key: new_polyhedron.first)
	//       std::cout << key<<' ';
	//     std::cout << "   ";
	//   }
	//   std::cout << '\n';
	//   for (const auto new_polyhedron: new_intersections)
	//   {
	//     for (const auto simplex: new_polyhedron.second)
	//       std::cout << tools::drawtriangle(simplex);
	//     std::cout <<"    ";
	//   }
	//   std::cout << '\n';

	//   //if (cut_part == 1) { PPause; }
	//   //if (cut_part == 0) { PPause; }
	// }

      	// Update before next stage
      	//all_intersections[stage] =
	previous_intersections = new_intersections;


	// Add quadrature rule with correct sign
	const double sign = std::pow(-1, stage);
        quadrature_rule overlap_part_qr;

	for (const auto polyhedron: new_intersections)
	  for (const auto simplex: polyhedron.second)
	  {
	    // if (std::abs(sign-1)<1e-10)
	    //   areapos += tools::area(simplex);
	    // else
	    //   areaminus += tools::area(simplex);
	    std::vector<double> x = convert(simplex, tdim, gdim);
	    _add_quadrature_rule(overlap_part_qr, x,
				 tdim, gdim, quadrature_order, sign);
	  }

        // Add quadrature rule for overlap part
        overlap_qr.push_back(overlap_part_qr);

	// {
	//   // Test the quadrature rule
	//   double volume = 0;
	//   for (std::size_t i = 0; i < overlap_part_qr.second.size(); ++i)
	//   {
	//     //std::cout << std::setprecision(20) << qr.second[i]<<'\n';
	//     volume += overlap_part_qr.second[i];
	//   }
	//   std::cout << "volume="<<volume<<'\n';
	//   //if (cut_part == 1) { PPause; }
	// }

	//PPause;
      }



      // std::cout << "\n summarize all intersections for part=" << cut_part<< " (there are "<<all_intersections.size()<< " stages)" << std::endl;

      // for (std::size_t stage = 0; stage < all_intersections.size(); ++stage)
      // {
      // 	std::cout << "\nstage " << stage << " has " << all_intersections[stage].size() << " polyhedra: " << std::endl
      // 		  << "figure("<<stage+1<<"),title('stage="<<stage<<"'),axis equal,hold on;\n";
      // 	for (const auto polyhedron: all_intersections[stage])
      // 	{
      // 	  for (const auto simplex: polyhedron.second)
      // 	    std::cout << tools::drawtriangle(simplex);

      // 	  // for (const auto key: polyhedron.first)
      // 	  //   std::cout << key <<' ';
      // 	  // std::cout << "    ";
      // 	}
      // }
      // std::cout << std::endl;


      // // qr is pair of point and weight (each is a vector<double>)
      // for (const auto qr: overlap_qr)
      // 	for (std::size_t i = 0; i < qr.second.size(); ++i)
      // 	{
      // 	  const Point pt(qr.first[gdim*i],qr.first[gdim*i+1]);
      // 	  if (qr.second[i] < 0)
      // 	    std::cout << tools::matlabplot(pt,"'o'");
      // 	  else
      // 	    std::cout << tools::matlabplot(pt,"'.'");
      // 	  std::cout << std::endl;
      // 	}


      // PPause;




      // Store quadrature rules for cut cell
      _quadrature_rules_overlap[cut_part][cut_cell_index] = overlap_qr;
      //_quadrature_rules_interface[cut_part][cut_cell_index] = interface_qr;

      // Store facet normals for cut cell
      //_facet_normals[cut_part][cut_cell_index] = interface_n;

    }

    // {
    //   // sum
    //   double part_volume = 0;
    //   double volume = 0;

    //   // Uncut cell volume given by function volume
    //   const auto uncut_cells = this->uncut_cells(cut_part);
    //   for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
    //   {
    // 	const Cell cell(*part(cut_part), *it);
    // 	volume += cell.volume();
    // 	part_volume += cell.volume();
    //   }

    //   // Cut cell volume given by quadrature rule
    //   const auto& cut_cells = this->cut_cells(cut_part);
    //   for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
    //   {
    // 	//const auto& qr = quadrature_rule_cut_cell(cut_part, *it);
    // 	const auto& overlap_qr = _quadrature_rules_overlap[cut_part][*it];
    // 	for (const auto qr: overlap_qr)
    // 	  for (std::size_t i = 0; i < qr.second.size(); ++i)
    // 	  {
    // 	    volume += qr.second[i];
    // 	    part_volume += qr.second[i];
    // 	  }
    //   }

    //   //std::cout<<" volumes "<<std::setprecision(16) << volume <<' ' << part_volume << ' ' <<(areapos-areaminus)<<'\n';

    //   // std::cout << "area" << cut_part << " = " << std::setprecision(15)<<areapos << " - " << areaminus << ",   1-("<<areapos<<"-"<<areaminus<<")"<<std::endl;

    //   double err=0;
    //   if (cut_part == num_parts()-1)
    // 	err = 0.36-part_volume;
    //   else
    // 	err = (areapos-areaminus)-part_volume;
    //   std::cout << "error " << cut_part << " " << err << '\n';
    // }

  }









  // // Iterate over all parts
  // for (std::size_t cut_part = 0; cut_part < num_parts(); cut_part++)
  // {
  //   // Iterate over cut cells for current part
  //   const auto& cmap = collision_map_cut_cells(cut_part);
  //   for (auto it = cmap.begin(); it != cmap.end(); ++it)
  //   {
  //     // Get cut cell
  //     const unsigned int cut_cell_index = it->first;
  //     const Cell cut_cell(*(_meshes[cut_part]), cut_cell_index);

  //     // Get dimensions
  //     const std::size_t tdim = cut_cell.mesh().topology().dim();
  //     const std::size_t gdim = cut_cell.mesh().geometry().dim();

  //     // Data structure for the volume triangulation of the cut_cell
  //     std::vector<double> volume_triangulation;

  //     // Data structure for the overlap quadrature rule
  //     std::vector<quadrature_rule> overlap_qr;

  //     // Data structure for the interface quadrature rule
  //     std::vector<quadrature_rule> interface_qr;

  //     // Data structure for the facet normals of the interface. The
  //     // numbering matches the numbering of interface_qr. This means
  //     // we have one normal for each quadrature point, since this is
  //     // how the data are grouped during assembly: for each pair of
  //     // colliding cells, we build a list of quadrature points and a
  //     // corresponding list of facet normals.
  //     std::vector<std::vector<double>> interface_n;

  //     // Data structure for the interface triangulation
  //     std::vector<double> interface_triangulation;

  //     // Data structure for normals to the interface. The numbering
  //     // should match the numbering of interface_triangulation.
  //     std::vector<Point> triangulation_normals;

  //     // Iterate over cutting cells
  //     const auto& cutting_cells = it->second;
  //     for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
  //     {
  //       // Get cutting part and cutting cell
  //       const std::size_t cutting_part = jt->first;
  //       const std::size_t cutting_cell_index = jt->second;
  //       const Cell cutting_cell(*(_meshes[cutting_part]), cutting_cell_index);

  //       // Topology of this cut part
  //       const std::size_t tdim_boundary = _boundary_meshes[cutting_part]->topology().dim();

  //       // Must have the same topology at the moment (FIXME)
  //       dolfin_assert(cutting_cell.mesh().topology().dim() == tdim);

  //       // Data structure for local interface triangulation
  //       std::vector<double> local_interface_triangulation;

  //       // Data structure for the local interface normals. The
  //       // numbering should match the numbering of
  //       // local_interface_triangulation.
  //       std::vector<Point> local_triangulation_normals;

  //       // Data structure for the overlap part quadrature rule
  //       quadrature_rule overlap_part_qr;

  //       // Data structure for the interface part quadrature rule
  //       quadrature_rule interface_part_qr;

  //       // Data structure for the interface part facet normals. The
  //       // numbering matches the numbering of interface_part_qr.
  //       std::vector<double> interface_part_n;

  //       // Iterate over boundary cells
  //       for (auto boundary_cell_index : full_to_bdry[cutting_part][cutting_cell_index])
  //       {
  //         // Get the boundary facet as a cell in the boundary mesh
  //         const Cell boundary_cell(*_boundary_meshes[cutting_part],
  //                                  boundary_cell_index.first);

  //         // Get the boundary facet as a facet in the full mesh
  //         const Facet boundary_facet(*_meshes[cutting_part],
  //                                    boundary_cell_index.second);

  //         // Triangulate intersection of cut cell and boundary cell
  //         const auto triangulation_cut_boundary
  //           = cut_cell.triangulate_intersection(boundary_cell);

  //         // The normals to triangulation_cut_boundary
  //         std::vector<Point> normals_cut_boundary;

  //         // Add quadrature rule and normals for triangulation
  //         if (triangulation_cut_boundary.size())
  //         {
  //           dolfin_assert(interface_part_n.size() == interface_part_qr.first.size());

  //           const auto num_qr_points
  //             = _add_quadrature_rule(interface_part_qr,
  //                                    triangulation_cut_boundary,
  //                                    tdim_boundary, gdim,
  //                                    quadrature_order, 1);

  //           const std::size_t local_facet_index = cutting_cell.index(boundary_facet);
  //           const Point n = -cutting_cell.normal(local_facet_index);
  //           for (std::size_t i = 0; i < num_qr_points.size(); ++i)
  //           {
  //             _add_normal(interface_part_n,
  //                         n,
  //                         num_qr_points[i],
  //                         gdim);
  //             normals_cut_boundary.push_back(n);
  //           }

  //           dolfin_assert(interface_part_n.size() == interface_part_qr.first.size());
  //         }

  //         // Triangulate intersection of boundary cell and previous volume triangulation
  //         const auto triangulation_boundary_prev_volume
  //           = IntersectionTriangulation::triangulate_intersection(boundary_cell,
  //                                                                 volume_triangulation,
  //                                                                 tdim);

  //         // Add quadrature rule and normals for triangulation
  //         if (triangulation_boundary_prev_volume.size())
  //         {
  //           dolfin_assert(interface_part_n.size() == interface_part_qr.first.size());

  //           const auto num_qr_points
  //             = _add_quadrature_rule(interface_part_qr,
  //                                    triangulation_boundary_prev_volume,
  //                                    tdim_boundary, gdim,
  //                                    quadrature_order, -1);

  //           const std::size_t local_facet_index = cutting_cell.index(boundary_facet);
  //           const Point n = -cutting_cell.normal(local_facet_index);
  //           for (std::size_t i = 0; i < num_qr_points.size(); ++i)
  //             _add_normal(interface_part_n,
  //                         n,
  //                         num_qr_points[i],
  //                         gdim);

  //           dolfin_assert(interface_part_n.size() == interface_part_qr.first.size());
  //         }

  //         // Update triangulation
  //         local_interface_triangulation.insert(local_interface_triangulation.end(),
  //                                              triangulation_cut_boundary.begin(),
  //                                              triangulation_cut_boundary.end());

  //         // Update interface facet normals
  //         local_triangulation_normals.insert(local_triangulation_normals.end(),
  //                                            normals_cut_boundary.begin(),
  //                                            normals_cut_boundary.end());
  //       }

  //       // Triangulate the intersection of the previous interface
  //       // triangulation and the cutting cell (to remove)
  //       std::vector<double> triangulation_prev_cutting;
  //       std::vector<Point> normals_prev_cutting;
  //       IntersectionTriangulation::triangulate_intersection(cutting_cell,
  //                                                           interface_triangulation,
  //                                                           triangulation_normals,
  //                                                           triangulation_prev_cutting,
  //                                                           normals_prev_cutting,
  //                                                           tdim_boundary);

  //       // Add quadrature rule for triangulation
  //       if (triangulation_prev_cutting.size())
  //       {
  //         dolfin_assert(interface_part_n.size() == interface_part_qr.first.size());

  //         const auto num_qr_points
  //           = _add_quadrature_rule(interface_part_qr,
  //                                  triangulation_prev_cutting,
  //                                  tdim_boundary, gdim,
  //                                  quadrature_order, -1);

  //         for (std::size_t i = 0; i < num_qr_points.size(); ++i)
  //           _add_normal(interface_part_n,
  //                       normals_prev_cutting[i],
  //                       num_qr_points[i],
  //                       gdim);

  //         dolfin_assert(interface_part_n.size() == interface_part_qr.first.size());
  //       }

  //       // Update triangulation
  //       interface_triangulation.insert(interface_triangulation.end(),
  //                                      local_interface_triangulation.begin(),
  //                                      local_interface_triangulation.end());

  //       // Update normals
  //       triangulation_normals.insert(triangulation_normals.end(),
  //                                    local_triangulation_normals.begin(),
  //                                    local_triangulation_normals.end());

  //       // Do the volume segmentation

  //       // Compute volume triangulation of intersection of cut and cutting cells
  //       const auto triangulation_cut_cutting
  //         = cut_cell.triangulate_intersection(cutting_cell);

  //       // Compute triangulation of intersection of cutting cell and
  //       // the (previous) volume triangulation
  //       const auto triangulation_cutting_prev
  //         = IntersectionTriangulation::triangulate_intersection(cutting_cell,
  //                                                               volume_triangulation,
  //                                                               tdim);

  //       // Add these new triangulations
  //       volume_triangulation.insert(volume_triangulation.end(),
  //                                   triangulation_cut_cutting.begin(),
  //                                   triangulation_cut_cutting.end());

  //       // Add quadrature rule with weights corresponding to the two
  //       // triangulations
  //       _add_quadrature_rule(overlap_part_qr,
  //                            triangulation_cut_cutting,
  //                            tdim, gdim, quadrature_order, 1);
  //       _add_quadrature_rule(overlap_part_qr,
  //                            triangulation_cutting_prev,
  //                            tdim, gdim, quadrature_order, -1);

  //       // Add quadrature rule for overlap part
  //       overlap_qr.push_back(overlap_part_qr);

  //       // Add quadrature rule for interface part
  //       interface_qr.push_back(interface_part_qr);

  //       // Add facet normal for interface part
  //       interface_n.push_back(interface_part_n);
  //     }

  //     // Store quadrature rules for cut cell
  //     _quadrature_rules_overlap[cut_part][cut_cell_index] = overlap_qr;
  //     _quadrature_rules_interface[cut_part][cut_cell_index] = interface_qr;

  //     // Store facet normals for cut cell
  //     _facet_normals[cut_part][cut_cell_index] = interface_n;
  //   }
  // }

  end();
}
//-----------------------------------------------------------------------------
  void MultiMesh::_build_quadrature_rules_cut_cells()
{
  begin(PROGRESS, "Building quadrature rules of cut cells.");

  // FIXME: Do we want to check to make sure we
  // have the same order in the overlapping part?

  // Get quadrature order
  const std::size_t quadrature_order = parameters["quadrature_order"];

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
      auto qr = SimplexQuadrature::compute_quadrature_rule(cut_cell,
                                                           quadrature_order);

      // Get the quadrature rule for the overlapping part
      const auto& qr_overlap = _quadrature_rules_overlap[cut_part][cut_cell_index];

      // Add the quadrature rule for the overlapping part to the
      // quadrature rule of the cut cell with flipped sign
      for (std::size_t k = 0; k < qr_overlap.size(); k++)
        _add_quadrature_rule(qr, qr_overlap[k], gdim, -1);

      // Store quadrature rule for cut cell
      _quadrature_rules_cut_cells[cut_part][cut_cell_index] = qr;
    }
  }

  end();
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>
MultiMesh::_add_quadrature_rule(quadrature_rule& qr,
                                const std::vector<double>& triangulation,
                                std::size_t tdim,
                                std::size_t gdim,
                                std::size_t quadrature_order,
                                double factor) const
{
  // Iterate over simplices in triangulation
  const std::size_t offset = (tdim + 1)*gdim; // coordinates per simplex
  const std::size_t num_simplices = triangulation.size() / offset;
  std::vector<std::size_t> num_points(num_simplices);

  for (std::size_t k = 0; k < num_simplices; k++)
  {
    // Get coordinates for current simplex in triangulation
    const double* x = &triangulation[0] + k*offset;

    // Compute quadrature rule for simplex
    const auto dqr = SimplexQuadrature::compute_quadrature_rule(x,
                                                               tdim,
                                                               gdim,
                                                               quadrature_order);

    // Add quadrature rule
    num_points[k] = _add_quadrature_rule(qr, dqr, gdim, factor);
  }
  return num_points;
}
//-----------------------------------------------------------------------------
std::size_t MultiMesh::_add_quadrature_rule(quadrature_rule& qr,
                                            const quadrature_rule& dqr,
                                            std::size_t gdim,
                                            double factor) const
{
  // Get the number of points
  dolfin_assert(dqr.first.size() == gdim*dqr.second.size());
  const std::size_t num_points = dqr.second.size();

  // Skip if sum of weights is too small
  double wsum = 0.0;
    for (std::size_t i = 0; i < num_points; i++)
    wsum += std::abs(dqr.second[i]);
  if (wsum < DOLFIN_EPS)
    return 0;

  // Append points and weights
  for (std::size_t i = 0; i < num_points; i++)
  {
    // Add point
    for (std::size_t j = 0; j < gdim; j++)
      qr.first.push_back(dqr.first[i*gdim + j]);

    // Add weight
    qr.second.push_back(factor*dqr.second[i]);
    //std::cout << std::setprecision(20) <<factor*dqr.second[i] << '\n';
  }

  return num_points;
}
//-----------------------------------------------------------------------------
 void MultiMesh::_add_normal(std::vector<double>& normals,
                             const Point& normal,
                             std::size_t npts,
                             std::size_t gdim) const
 {
   for (std::size_t i = 0; i < npts; ++i)
     for (std::size_t j = 0; j < gdim; ++j)
       normals.push_back(normal[j]);
 }

//-----------------------------------------------------------------------------
void MultiMesh::_plot() const
{
  cout << "Plotting multimesh with " << num_parts() << " parts" << endl;

  // Iterate over parts
  for (std::size_t p = 0; p < num_parts(); ++p)
  {
    // Create a cell function and mark cells
    std::shared_ptr<MeshFunction<std::size_t>>
      f(new MeshFunction<std::size_t>(part(p),
                                      part(p)->topology().dim()));

    // Set all entries to 0 (uncut cells)
    f->set_all(0);

    // Mark cut cells as 1
    for (auto it : cut_cells(p))
      f->set_value(it, 1);

    // Mart covered cells as 2
    for (auto it : covered_cells(p))
      f->set_value(it, 2);

    // Write some debug data
    const std::size_t num_cut = cut_cells(p).size();
    const std::size_t num_covered = covered_cells(p).size();
    const std::size_t num_uncut = part(p)->num_cells() - num_cut - num_covered;
    cout << "Part " << p << " has "
         << num_uncut   << " uncut cells (0), "
         << num_cut     << " cut cells (1), and "
         << num_covered << " covered cells (2)." << endl;

    // Plot
    std::stringstream s;
    s << "Map of cell types for multimesh part " << p;
    plot(f, s.str());
  }
}
//------------------------------------------------------------------------------
std::size_t MultiMesh::n_choose_k(std::size_t n,
				  std::size_t k)
{
  if (k == 0) return 1;
  return (n * n_choose_k(n - 1, k - 1)) / k;
}
//-----------------------------------------------------------------------------
std::vector<std::deque<std::size_t> >
MultiMesh::compute_permutations(std::size_t n,
				std::size_t k)
{
  // This is slow for large n and k

  switch (k)
  {
  case 0:
    return std::vector<std::deque<std::size_t> >();
  case 1:
    {
      std::vector<std::deque<std::size_t> > pp(n, std::deque<std::size_t>());
      for (std::size_t i = 0; i < n; i++)
	pp[i].push_back(i);
      return pp;
    }
  default:
    {
      std::vector<std::deque<std::size_t> > pp = compute_permutations(k - 1, n);
      std::vector<std::deque<std::size_t> > permutations;

      for (std::size_t i = 0; i < n; ++i)
	for (const auto& p: pp)
	  if (i < p[0])
	  {
	    std::deque<std::size_t> q = p;
	    q.push_front(i);
	    permutations.push_back(q);
	  }
      return permutations;
    }
  }
}
//------------------------------------------------------------------------------
double MultiMesh::minimum_angle(const Simplex& s) const
{
  const double a2 = (s[1]-s[0]).squared_norm();
  const double b2 = (s[2]-s[0]).squared_norm();
  const double c2 = (s[2]-s[1]).squared_norm();

  // Cosine thrm
  const double alpha = acos(0.5 * (b2 + c2 - a2) / std::sqrt(b2 * c2));
  const double beta = acos(0.5 * (a2 + c2 - b2) / std::sqrt(a2 * c2));

  std::cout << "angles " << alpha <<' '<<beta<<' '<<DOLFIN_PI - (alpha + beta) <<'\n';

  double min_angle = alpha;
  min_angle = std::min(min_angle, beta);
  min_angle = std::min(min_angle, DOLFIN_PI - (alpha + beta));

  return min_angle;
}

//------------------------------------------------------------------------------
bool MultiMesh::maximize_minimum_angle(Polyhedron& polyhedron) const
{
  return false;

  double min_angle = DOLFIN_PI;
  static const double angle_tol = 1e-3;
  int trino = -1;
  double initial_area = 0;

  for (const auto tri: polyhedron)
  {
    //min_angle = std::min(min_angle, minimum_angle(tri));
    const double v = minimum_angle(tri);
    if (v < min_angle) {
      min_angle = v;
      trino++;
    }

    initial_area += tools::area(tri);

    // if (min_angle < angle_tol)
    // {
    //   std::cout << min_angle << '\n'
    // 		<< tools::drawtriangle(tri) << '\n'
    // 		<< a2 << ' ' << b2<<' ' << c2 << ' ' << alpha << ' ' << beta<<'\n';

    //   PPause;
    // }

  }

  // Flip if angle is small and there are at least two triangles in polyhedron
  std::cout << "min_angle = " << min_angle << " at trino " << trino << '\n';

  if (min_angle < angle_tol)
  {
    for (const auto tri: polyhedron)
      std::cout << tools::drawtriangle(tri);
    std::cout << std::endl;

    // Must be at least two triangles
    if (polyhedron.size() > 1)
    {
      if (trino == 0)// or trino == 2)
      {
	Polyhedron new_polyhedron(polyhedron.size(),
				  std::vector<Point>(3));
	new_polyhedron[trino][0] = polyhedron[trino][0];
	new_polyhedron[trino][1] = polyhedron[trino][1];
	new_polyhedron[trino][2] = polyhedron[1][2];
	new_polyhedron[1][0] = polyhedron[trino][1];
	new_polyhedron[1][1] = polyhedron[trino][2];
	new_polyhedron[1][2] = polyhedron[1][2];
	for (std::size_t i = 2; i < polyhedron.size(); ++i)
	  for (std::size_t j = 0; j < 3; ++j)
	    new_polyhedron[i][j] = polyhedron[i][j];

	polyhedron = new_polyhedron;

	for (const auto s: polyhedron)
	  std::cout << tools::drawtriangle(s);
	std::cout << '\n';
	for (const auto s: polyhedron)
	  std::cout << tools::area(s)<<' ';
	std::cout << '\n';

      }
      else
      {
	PPause;
      }

      double post_area = 0;
      for (const auto tri: polyhedron)
	post_area += tools::area(tri);

      if (std::abs(post_area-initial_area) > 1e-13)
      {
	std::cout<<std::setprecision(15) << "area error " << post_area << ' ' << initial_area << ' '<<std::abs(post_area-initial_area)<<std::endl;
	exit(0);
      }

      return true;
    }

  }

  return false;

}
//------------------------------------------------------------------------------
