// Copyright (C) 2010 Anders Logg
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
// Modified by Garth N. Wells, 2011.
//
// First added:  2010-11-27
// Last changed: 2014-02-06

#include <algorithm>
#include <vector>

#include <dolfin/log/log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include "Cell.h"
#include "Mesh.h"
#include "MeshEditor.h"
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshRenumbering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh MeshRenumbering::renumber_by_color(const Mesh& mesh,
                                 const std::vector<std::size_t> coloring_type)
{
  // Start timer
  Timer timer("Renumber mesh by color");

  // Get some some mesh
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t num_local_vertices  = mesh.size(0);
  const std::size_t num_global_vertices = mesh.size_global(0);
  const std::size_t num_local_cells     = mesh.size(tdim);
  const std::size_t num_global_cells    = mesh.size_global(tdim);

  // Check that requested coloring is a cell coloring
  if (coloring_type[0] != tdim)
  {
    dolfin_error("MeshRenumbering.cpp",
                 "renumber mesh by color",
                 "Coloring is not a cell coloring: only cell colorings are supported");
  }

  // Compute renumbering
  std::vector<double> new_coordinates;
  std::vector<std::size_t> new_connections;
  MeshRenumbering::compute_renumbering(mesh, coloring_type, new_coordinates,
                                       new_connections);

  // Create new mesh
  Mesh new_mesh;

  // Create mesh editor
  MeshEditor editor;
  editor.open(new_mesh, mesh.type().cell_type(), tdim, gdim);
  editor.init_cells_global(num_local_cells, num_global_cells);
  editor.init_vertices_global(num_local_vertices, num_global_vertices);

  // Add vertices
  dolfin_assert(new_coordinates.size() == num_local_vertices*gdim);
  for (std::size_t i = 0; i < num_local_vertices; ++i)
  {
    std::vector<double> x(gdim);
    for (std::size_t j = 0; j < gdim; ++j)
      x[j] = new_coordinates[i*gdim + j];
    editor.add_vertex(i, x);
  }

  cout << "Done adding vertices" << endl;

  // Add cells
  dolfin_assert(new_coordinates.size() == num_local_vertices*gdim);
  const std::size_t vertices_per_cell = mesh.type().num_entities(0);
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    std::vector<std::size_t> c(vertices_per_cell);
    std::copy(new_connections.begin() + i*vertices_per_cell,
              new_connections.begin() + i*vertices_per_cell + vertices_per_cell,
              c.begin());
    editor.add_cell(i, c);
  }

  editor.close();

  cout << "Close editor" << endl;

  // Initialise coloring data
  typedef std::map<std::vector<std::size_t>, std::pair<std::vector<std::size_t>,
           std::vector<std::vector<std::size_t>>>>::const_iterator
    ConstMeshColoringData;

  // Get old coloring
  ConstMeshColoringData mesh_coloring
    = mesh.topology().coloring.find(coloring_type);
  if (mesh_coloring == mesh.topology().coloring.end())
  {
    dolfin_error("MeshRenumbering.cpp",
                 "renumber mesh by color",
                 "Requested mesh coloring has not been computed");
  }

  // Get old coloring data
  const std::vector<std::size_t>& colors = mesh_coloring->second.first;
  const std::vector<std::vector<std::size_t>>&
    entities_of_color = mesh_coloring->second.second;
  dolfin_assert(colors.size() == num_local_cells);
  dolfin_assert(!entities_of_color.empty());
  const std::size_t num_colors = entities_of_color.size();

  // New coloring data
  dolfin_assert(new_mesh.topology().coloring.empty());
  std::vector<std::size_t> new_colors(colors.size());
  std::vector<std::vector<std::size_t>> new_entities_of_color(num_colors);

  std::size_t current_cell = 0;
  for (std::size_t color = 0; color < num_colors; color++)
  {
    // Get the array of cell indices of current color
    const std::vector<std::size_t>& colored_cells = entities_of_color[color];
    std::vector<std::size_t>& new_colored_cells = new_entities_of_color[color];

    // Update cell color data
    for (std::size_t i = 0; i < colored_cells.size(); i++)
    {
      new_colored_cells.push_back(current_cell);
      new_colors[current_cell] = color;
      current_cell++;
    }
  }

  // Set new coloring mesh data
  std::pair<ConstMeshColoringData, bool> insert
    = new_mesh.topology().coloring.insert(std::make_pair(coloring_type,
                          std::make_pair(new_colors, new_entities_of_color)));
  dolfin_assert(insert.second);

  cout << "Return new mesh" << endl;
  return new_mesh;
}
//-----------------------------------------------------------------------------
void MeshRenumbering::compute_renumbering(const Mesh& mesh,
                                          const std::vector<std::size_t>& coloring_type,
                                          std::vector<double>& new_coordinates,
                                          std::vector<std::size_t>& new_connections)
{
  // Get some some mesh
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t num_vertices = mesh.num_vertices();
  const std::size_t num_cells = mesh.num_cells();

  // Resize vectors
  const MeshConnectivity& connectivity = mesh.topology()(tdim, 0);
  const std::size_t connections_size = connectivity.size();
  new_connections.resize(connections_size);

  const std::size_t coordinates_size
    = mesh.geometry().num_vertices()*mesh.geometry().dim();
  new_coordinates.resize(coordinates_size);

  typedef std::map<std::vector<std::size_t>, std::pair<std::vector<std::size_t>,
           std::vector<std::vector<std::size_t>>>>::const_iterator MeshColoringData;

  info("Renumbering mesh by cell colors.");
  info(mesh);

  // Check that requested coloring is a cell coloring
  if (coloring_type[0] != mesh.topology().dim())
  {
    dolfin_error("MeshRenumbering.cpp",
                 "compute renumbering of mesh",
                 "Coloring is not a cell coloring: only cell colorings are supported");
  }

  // Get coloring
  MeshColoringData mesh_coloring = mesh.topology().coloring.find(coloring_type);

  // Check that requested coloring has been computed
  if (mesh_coloring == mesh.topology().coloring.end())
  {
    dolfin_error("MeshRenumbering.cpp",
                 "compute renumbering of mesh",
                 "Requested mesh coloring has not been computed");
  }

  // Get coloring data
  const std::vector<std::size_t>& colors_old = mesh_coloring->second.first;
  const std::vector<std::vector<std::size_t>>&
    entities_of_color_old = mesh_coloring->second.second;
  dolfin_assert(colors_old.size() == num_cells);
  dolfin_assert(!entities_of_color_old.empty());

  // Get coordinates
  const std::vector<double>& coordinates = mesh.geometry().x();

  // New vertex indices, -1 if not yet renumbered
  std::vector<int> new_vertex_indices(num_vertices, -1);

  // Iterate over colors
  const std::size_t num_colors = entities_of_color_old.size();
  std::size_t connections_offset = 0;
  std::size_t current_vertex = 0;
  for (std::size_t color = 0; color < num_colors; ++color)
  {
    // Get the array of cell indices of current color
    const std::vector<std::size_t>& colored_cells = entities_of_color_old[color];

    // Iterate over cells for current color
    for (std::size_t i = 0; i < colored_cells.size(); i++)
    {
      // Current cell
      Cell cell(mesh, colored_cells[i]);

      // Get array of vertices for current cell
      const unsigned int* cell_vertices = cell.entities(0);
      const std::size_t num_cell_vertices   = cell.num_entities(0);

      // Iterate over cell vertices
      for (std::size_t j = 0; j < num_cell_vertices; j++)
      {
        // Get vertex index
        const std::size_t vertex_index = cell_vertices[j];

        // Renumber and copy coordinate data if vertex is not yet renumbered
        if (new_vertex_indices[vertex_index] == -1)
        {
          std::copy(coordinates.begin() + vertex_index*gdim,
                    coordinates.begin() + (vertex_index + 1)*gdim,
                    new_coordinates.begin() + current_vertex*gdim);
          new_vertex_indices[vertex_index] = current_vertex++;
        }

        // Renumber and copy connectivity data (must be done after vertex renumbering)
        const std::size_t new_vertex_index = new_vertex_indices[vertex_index];
        new_connections[connections_offset++] = new_vertex_index;
      }
    }
  }

  // Check that all vertices have been renumbered
  for (std::size_t i = 0; i < new_vertex_indices.size(); i++)
  {
    if (new_vertex_indices[i] == -1)
    {
      dolfin_error("MeshRenumbering.cpp",
                   "compute renumbering of mesh",
                   "Some vertices were not renumbered");
    }
  }
}
//-----------------------------------------------------------------------------
