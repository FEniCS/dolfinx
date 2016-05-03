// Copyright (C) 2006-2013 Anders Logg and Garth N. Wells
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
// Modified by Johan Jansson 2006.
// Modified by Ola Skavhaug 2006.
// Modified by Niclas Jansson 2009.
// Modified by Oeyvind Evju, 2013
//
// First added:  2006-06-21
// Last changed: 2014-02-06

#include <dolfin/log/log.h>
#include "BoundaryMesh.h"
#include "Cell.h"
#include "Facet.h"
#include "Mesh.h"
#include "MeshData.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshGeometry.h"
#include "MeshTopology.h"
#include "Vertex.h"
#include "BoundaryComputation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void BoundaryComputation::compute_boundary(const Mesh& mesh,
                                           const std::string type,
                                           BoundaryMesh& boundary)
{
  // We iterate over all facets in the mesh and check if they are on
  // the boundary. A facet is on the boundary if it is connected to
  // exactly one cell.

  log(TRACE, "Computing boundary mesh.");

  bool exterior = true;
  bool interior = true;
  if (type == "exterior")
    interior = false;
  else if (type == "interior")
    exterior = false;
  else if (type != "local")
  {
    dolfin_error("BoundaryComputation.cpp",
                 "determine boundary mesh type",
                 "Unknown boundary type (%d)", type.c_str());
  }

  // Get my MPI process rank and number of MPI processes
  const std::size_t my_rank = MPI::rank(mesh.mpi_comm());
  const std::size_t num_processes = MPI::size(mesh.mpi_comm());

  // Open boundary mesh for editing
  const std::size_t D = mesh.topology().dim();
  MeshEditor editor;
  editor.open(boundary, mesh.type().facet_type(), D - 1, mesh.geometry().dim());

  // Generate facet - cell connectivity if not generated
  mesh.init(D - 1, D);

  // Temporary arrays for assignment of indices to vertices on the boundary
  std::map<std::size_t, std::size_t> boundary_vertices;

  // Map of index "owners" (process responsible for assigning global index)
  std::map< std::size_t, std::size_t > global_index_owner;

  // Shared vertices for full mesh
  // FIXME: const_cast
  const std::map<unsigned int, std::set<unsigned int>> &
    shared_vertices = const_cast<Mesh&>(mesh).topology().shared_entities(0);

  // Shared vertices for boundary mesh
  std::map<unsigned int, std::set<unsigned int>> shared_boundary_vertices;
  if (exterior)
  {
    // Extract shared vertices if vertex is identified as part of globally
    // exterior facet.
    std::vector<std::size_t> boundary_global_indices;
    for (std::map<unsigned int, std::set<unsigned int>>::const_iterator
        sv_it=shared_vertices.begin(); sv_it != shared_vertices.end(); ++sv_it)
    {
      std::size_t local_mesh_index = sv_it->first;
      Vertex v(mesh, local_mesh_index);

      for (FacetIterator f(v); !f.end(); ++f)
      {
        if (f->num_global_entities(D) == 1)
        {
          const std::size_t global_mesh_index
            = mesh.topology().global_indices(0)[local_mesh_index];
          shared_boundary_vertices[local_mesh_index] = sv_it->second;
          boundary_global_indices.push_back(global_mesh_index);
          break;
        }
      }
    }

    // Distribute all shared boundary vertices
    std::vector<std::vector<std::size_t>> boundary_global_indices_all;
    MPI::all_gather(mesh.mpi_comm(), boundary_global_indices,
                     boundary_global_indices_all);

    // Identify and clean up discrepancies between shared vertices of full mesh
    // and shared vertices of boundary mesh
    for (auto sbv_it = shared_boundary_vertices.begin();
         sbv_it != shared_boundary_vertices.end(); )
    {
      std::size_t local_mesh_index = sbv_it->first;
      const std::size_t global_mesh_index
        = mesh.topology().global_indices(0)[local_mesh_index];

      // Check if this vertex is identified as boundary vertex on
      // other processes sharing this vertex
      std::set<unsigned int> &other_processes = sbv_it->second;
      for (auto  op_it=other_processes.begin();
           op_it != other_processes.end(); )
      {
        // Check if vertex is identified as boundary vertex on process *op_it
        bool is_boundary_vertex
          = (std::find(boundary_global_indices_all[*op_it].begin(),
                      boundary_global_indices_all[*op_it].end(),
                      global_mesh_index)
             != boundary_global_indices_all[*op_it].end());

        // Erase item if this is not identified as a boundary vertex
        // on process *op_it, and increment iterator
        if (!is_boundary_vertex)
        {
          // Erase item while carefully avoiding invalidating the
          // iterator: First increment it to get the next, valid
          // iterator, and then erase what it pointed to from
          // other_processes
          other_processes.erase(op_it++);
        }
        else
          ++op_it;
      }

      // Erase item from map if no other processes identify this
      // vertex as a boundary vertex, and increment iterator
      if (other_processes.size() == 0)
      {
        // Erase carefully as above
        shared_boundary_vertices.erase(sbv_it++);
      }
      else
        ++sbv_it;
    }
  }
  else
  {
    // If interior boundary, shared vertices are the same
    shared_boundary_vertices = shared_vertices;
  }

  // Determine boundary facet, count boundary vertices and facets, and
  // assign vertex indices
  std::size_t num_boundary_vertices = 0;
  std::size_t num_owned_vertices = 0;
  std::size_t num_boundary_cells = 0;

  MeshFunction<bool> boundary_facet(reference_to_no_delete_pointer(mesh),
                                    D - 1, false);
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    // Boundary facets are connected to exactly one cell
    if (f->num_entities(D) == 1)
    {
      const bool global_exterior_facet =  (f->num_global_entities(D) == 1);
      if (global_exterior_facet && exterior)
        boundary_facet[*f] = true;
      else if (!global_exterior_facet && interior)
        boundary_facet[*f] = true;

      if (boundary_facet[*f])
      {
        // Count boundary vertices and assign indices
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          const std::size_t local_mesh_index = v->index();

          if (boundary_vertices.find(local_mesh_index)
              == boundary_vertices.end())
          {
            const std::size_t local_boundary_index = num_boundary_vertices;
            boundary_vertices[local_mesh_index] = local_boundary_index;

            // Determine "owner" of global_mesh_index
            std::size_t owner = my_rank;

            std::map<unsigned int, std::set<unsigned int>>::const_iterator
              other_processes_it
              = shared_boundary_vertices.find(local_mesh_index);
            if (other_processes_it != shared_boundary_vertices.end() && D > 1)
            {
              const std::set<unsigned int>& other_processes
                = other_processes_it->second;
              const std::size_t min_process
                = *std::min_element(other_processes.begin(),
                                    other_processes.end());
              boundary.topology().shared_entities(0)[local_boundary_index]
                = other_processes;

              // FIXME: More sophisticated ownership determination
              if (min_process < owner)
                owner = min_process;
            }
            const std::size_t global_mesh_index
              = mesh.topology().global_indices(0)[local_mesh_index];
            global_index_owner[global_mesh_index] = owner;

            // Update counts
            if (owner == my_rank)
              num_owned_vertices++;
            num_boundary_vertices++;
          }
        }

        // Count boundary cells (facets of the mesh)
        num_boundary_cells++;
      }
    }
  }

  // Initiate boundary topology
  /*
  boundary.topology().init(0, num_boundary_vertices,
                           MPI::sum(mesh.mpi_comm(), num_owned_vertices));
  boundary.topology().init(D - 1, num_boundary_cells,
                           MPI::sum(mesh.mpi_comm(), num_boundary_cells));
  */

  // Specify number of vertices and cells
  editor.init_vertices_global(num_boundary_vertices,
                              MPI::sum(mesh.mpi_comm(), num_owned_vertices));
  editor.init_cells_global(num_boundary_cells, MPI::sum(mesh.mpi_comm(),
                                                        num_boundary_cells));

  // Write vertex map
  MeshFunction<std::size_t>& vertex_map = boundary.entity_map(0);
  if (num_boundary_vertices > 0)
  {
    vertex_map.init(reference_to_no_delete_pointer(boundary), 0,
                    num_boundary_vertices);
  }
  std::map<std::size_t, std::size_t>::const_iterator it;
  for (it = boundary_vertices.begin(); it != boundary_vertices.end(); ++it)
    vertex_map[it->second] = it->first;

  // Get vertex ownership distribution, and find index to start global
  // numbering from
  std::vector<std::size_t> ownership_distribution(num_processes);
  MPI::all_gather(mesh.mpi_comm(), num_owned_vertices, ownership_distribution);
  std::size_t start_index = 0;
  for (std::size_t j = 0; j < my_rank; j++)
    start_index += ownership_distribution[j];

  // Set global indices of owned vertices, request global indices for
  // vertices owned elsewhere
  std::map<std::size_t, std::size_t> global_indices;
  std::vector<std::vector<std::size_t>> request_global_indices(num_processes);

  std::size_t current_index = start_index;
  for (std::size_t local_boundary_index = 0;
       local_boundary_index<num_boundary_vertices; local_boundary_index++)
  {
    const std::size_t local_mesh_index = vertex_map[local_boundary_index];
    const std::size_t global_mesh_index
      = mesh.topology().global_indices(0)[local_mesh_index];

    const std::size_t owner = global_index_owner[global_mesh_index];
    if (owner != my_rank)
      request_global_indices[owner].push_back(global_mesh_index);
    else
      global_indices[global_mesh_index] = current_index++;
  }

  // Send and receive requests from other processes
  std::vector<std::vector<std::size_t>> global_index_requests(num_processes);
  MPI::all_to_all(mesh.mpi_comm(), request_global_indices,
                  global_index_requests);

  // Find response to requests of global indices
  std::vector<std::vector<std::size_t>> respond_global_indices(num_processes);
  for (std::size_t i = 0; i < num_processes; i++)
  {
    const std::size_t N = global_index_requests[i].size();
    respond_global_indices[i].resize(N);

    for (std::size_t j = 0; j < N; j++)
      respond_global_indices[i][j]
        = global_indices[global_index_requests[i][j]];
  }

  // Scatter responses back to requesting processes
  std::vector<std::vector<std::size_t>> global_index_responses(num_processes);
  MPI::all_to_all(mesh.mpi_comm(), respond_global_indices,
                  global_index_responses);

  // Update global_indices
  for (std::size_t i = 0; i < num_processes; i++)
  {
    const std::size_t N = global_index_responses[i].size();
    // Check that responses are the same size as the requests made
    dolfin_assert(global_index_responses[i].size()
                  == request_global_indices[i].size());
    for (std::size_t j = 0; j < N; j++)
    {
      const std::size_t global_mesh_index = request_global_indices[i][j];
      const std::size_t global_boundary_index = global_index_responses[i][j];
      global_indices[global_mesh_index] = global_boundary_index;
    }
  }

  // Create vertices
  for (std::size_t local_boundary_index = 0;
       local_boundary_index < num_boundary_vertices; local_boundary_index++)
  {
    const std::size_t local_mesh_index = vertex_map[local_boundary_index];
    const std::size_t global_mesh_index
      = mesh.topology().global_indices(0)[local_mesh_index];
    const std::size_t global_boundary_index = global_indices[global_mesh_index];

    Vertex v(mesh, local_mesh_index);

    editor.add_vertex_global(local_boundary_index, global_boundary_index,
                             v.point());
  }

  // Find global index to start cell numbering from for current process
  std::vector<std::size_t> cell_distribution(num_processes);
  MPI::all_gather(mesh.mpi_comm(), num_boundary_cells, cell_distribution);
  std::size_t start_cell_index = 0;
  for (std::size_t i = 0; i < my_rank; i++)
    start_cell_index += cell_distribution[i];

  // Create cells (facets) and map between boundary mesh cells and facets parent
  MeshFunction<std::size_t>& cell_map = boundary.entity_map(D - 1);
  if (num_boundary_cells > 0)
  {
    cell_map.init(reference_to_no_delete_pointer(boundary), D - 1,
                  num_boundary_cells);
  }
  std::vector<std::size_t>
    cell(boundary.type().num_vertices(boundary.topology().dim()));
  std::size_t current_cell = 0;
  for (FacetIterator f(mesh); !f.end(); ++f)
  {
    if (boundary_facet[*f])
    {
      // Compute new vertex numbers for cell
      const unsigned int* vertices = f->entities(0);
      for (std::size_t i = 0; i < cell.size(); i++)
        cell[i] = boundary_vertices[vertices[i]];

      // Reorder vertices so facet is right-oriented w.r.t. facet
      // normal
      reorder(cell, *f);

      // Create mapping from boundary cell to mesh facet if requested
      if (!cell_map.empty())
        cell_map[current_cell] = f->index();

      // Add cell
      editor.add_cell(current_cell, start_cell_index+current_cell, cell);
      current_cell++;
    }
  }

  // Close mesh editor. Note the argument order=false to prevent
  // ordering from destroying the orientation of facets accomplished
  // by calling reorder() below.
  editor.close(false);
}
//-----------------------------------------------------------------------------
void BoundaryComputation::reorder(std::vector<std::size_t>& vertices,
                                  const Facet& facet)
{
  // Get mesh
  const Mesh& mesh = facet.mesh();

  // Get the vertex opposite to the facet (the one we remove)
  std::size_t vertex = 0;
  const Cell cell(mesh, facet.entities(mesh.topology().dim())[0]);
  for (std::size_t i = 0; i < cell.num_entities(0); i++)
  {
    bool not_in_facet = true;
    vertex = cell.entities(0)[i];
    for (std::size_t j = 0; j < facet.num_entities(0); j++)
    {
      if (vertex == facet.entities(0)[j])
      {
        not_in_facet = false;
        break;
      }
    }
    if (not_in_facet)
      break;
  }
  const Point p = mesh.geometry().point(vertex);

  // Check orientation
  switch (mesh.type().cell_type())
  {
  case CellType::interval:
    // Do nothing
    break;
  case CellType::triangle:
    {
      dolfin_assert(facet.num_entities(0) == 2);

      const Point p0 = mesh.geometry().point(facet.entities(0)[0]);
      const Point p1 = mesh.geometry().point(facet.entities(0)[1]);
      const Point v = p1 - p0;
      const Point n(v.y(), -v.x());

      if (n.dot(p0 - p) < 0.0)
      {
        const std::size_t tmp = vertices[0];
        vertices[0] = vertices[1];
        vertices[1] = tmp;
      }
    }
    break;
  case CellType::tetrahedron:
    {
      dolfin_assert(facet.num_entities(0) == 3);

      const Point p0 = mesh.geometry().point(facet.entities(0)[0]);
      const Point p1 = mesh.geometry().point(facet.entities(0)[1]);
      const Point p2 = mesh.geometry().point(facet.entities(0)[2]);
      const Point v1 = p1 - p0;
      const Point v2 = p2 - p0;
      const Point n  = v1.cross(v2);

      if (n.dot(p0 - p) < 0.0)
      {
        const std::size_t tmp = vertices[0];
        vertices[0] = vertices[1];
        vertices[1] = tmp;
      }
    }
    break;
  default:
    {
      dolfin_error("BoundaryComputation.cpp",
                   "reorder cell for extraction of mesh boundary",
                   "Unknown cell type (%d)",
                   mesh.type().cell_type());
    }
  }
}
//-----------------------------------------------------------------------------
