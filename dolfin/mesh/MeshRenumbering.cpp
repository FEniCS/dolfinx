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
// Last changed: 2011-01-16

#include <algorithm>
#include <vector>
#include <boost/scoped_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include "Cell.h"
#include "Mesh.h"
#include "MeshEditor.h"
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshRenumbering.h"
#include "ParallelData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh
MeshRenumbering::renumber_by_color(const Mesh& mesh,
                                   const std::vector<unsigned int> coloring_type)
{
  // Start timer
  Timer timer("Renumber mesh by color");

  // Get some some mesh
  const uint tdim = mesh.topology().dim();
  const uint gdim = mesh.geometry().dim();
  const uint num_vertices = mesh.num_vertices();
  const uint num_cells = mesh.num_cells();

  // Check that requested coloring is a cell coloring
  if (coloring_type[0] != tdim)
    dolfin_error("MeshRenumbering.cpp",
                 "renumber mesh by color",
                 "Coloring is not a cell coloring: only cell colorings are supported");

  // Compute renumbering
  std::vector<double> new_coordinates;
  std::vector<uint> new_connections;
  MeshRenumbering::compute_renumbering(mesh, coloring_type, new_coordinates,
                                       new_connections);

  // Create new mesh
  Mesh new_mesh;

  // Create mesh editor
  MeshEditor editor;
  editor.open(new_mesh, mesh.type().cell_type(), tdim, gdim);
  editor.init_cells(num_cells);
  editor.init_vertices(num_vertices);

  // Add vertices
  dolfin_assert(new_coordinates.size() == num_vertices*gdim);
  for (uint i = 0; i < num_vertices; ++i)
  {
    const Point p(gdim, &new_coordinates[i*gdim]);
    editor.add_vertex(i, p);
  }

  // Add cells
  dolfin_assert(new_coordinates.size() == num_vertices*gdim);
  const uint vertices_per_cell = mesh.type().num_entities(0);
  for (uint i = 0; i < num_cells; ++i)
  {
    std::vector<uint> c(vertices_per_cell);
    std::copy(new_connections.begin() + i*vertices_per_cell,
              new_connections.begin() + i*vertices_per_cell + vertices_per_cell,
              c.begin());
    editor.add_cell(i, c);
  }

  editor.close();

  // Initialise coloring data
  typedef std::map<const std::vector<uint>, std::pair<MeshFunction<uint>,
           std::vector<std::vector<uint> > > >::const_iterator ConstMeshColoringData;

  // Get old coloring
  ConstMeshColoringData mesh_coloring
    = mesh.parallel_data().coloring.find(coloring_type);
  if (mesh_coloring == mesh.parallel_data().coloring.end())
    dolfin_error("MeshRenumbering.cpp",
                 "renumber mesh by color",
                 "Requested mesh coloring has not been computed");

  // Get old coloring data
  const MeshFunction<uint>& colors = mesh_coloring->second.first;
  const std::vector<std::vector<uint> >&
    entities_of_color = mesh_coloring->second.second;
  dolfin_assert(colors.size() == num_cells);
  dolfin_assert(!entities_of_color.empty());
  const uint num_colors = entities_of_color.size();

  // New coloring data
  dolfin_assert(new_mesh.parallel_data().coloring.empty());
  MeshFunction<uint> new_colors(mesh, tdim);
  std::vector<std::vector<uint> > new_entities_of_color(num_colors);

  uint current_cell = 0;
  for (uint color = 0; color < num_colors; color++)
  {
    // Get the array of cell indices of current color
    const std::vector<uint>& colored_cells = entities_of_color[color];
    std::vector<uint>& new_colored_cells = new_entities_of_color[color];

    // Update cell color data
    for (uint i = 0; i < colored_cells.size(); i++)
    {
      new_colored_cells.push_back(current_cell);
      new_colors[current_cell] = color;
      current_cell++;
    }
  }

  // Set new coloring mesh data
  std::pair<ConstMeshColoringData, bool> insert
    = new_mesh.parallel_data().coloring.insert(std::make_pair(coloring_type,
                          std::make_pair(new_colors, new_entities_of_color)));
  dolfin_assert(insert.second);

  return new_mesh;
}
//-----------------------------------------------------------------------------
void MeshRenumbering::compute_renumbering(const Mesh& mesh,
                                          const std::vector<dolfin::uint>& coloring_type,
                                          std::vector<double>& new_coordinates,
                                          std::vector<uint>& new_connections)
{
  // Get some some mesh
  const uint tdim = mesh.topology().dim();
  const uint gdim = mesh.geometry().dim();
  const uint num_vertices = mesh.num_vertices();
  const uint num_cells = mesh.num_cells();

  // Resize vectors
  const MeshConnectivity& connectivity = mesh.topology()(tdim, 0);
  const uint connections_size = connectivity.size();
  new_connections.resize(connections_size);

  const uint coordinates_size = mesh.geometry().size()*mesh.geometry().dim();
  new_coordinates.resize(coordinates_size);

  typedef std::map<const std::vector<uint>, std::pair<MeshFunction<uint>,
           std::vector<std::vector<uint> > > >::const_iterator MeshColoringData;

  info("Renumbering mesh by cell colors.");
  info(mesh);

  // Check that requested coloring is a cell coloring
  if (coloring_type[0] != mesh.topology().dim())
    dolfin_error("MeshRenumbering.cpp",
                 "compute renumbering of mesh",
                 "Coloring is not a cell coloring: only cell colorings are supported");

  // Get coloring
  MeshColoringData mesh_coloring = mesh.parallel_data().coloring.find(coloring_type);

  // Check that requested coloring has been computed
  if (mesh_coloring == mesh.parallel_data().coloring.end())
    dolfin_error("MeshRenumbering.cpp",
                 "compute renumbering of mesh",
                 "Requested mesh coloring has not been computed");

  // Get coloring data (copies since the data will be deleted mesh.clear())
  const MeshFunction<uint>& colors_old = mesh_coloring->second.first;
  const std::vector<std::vector<uint> >&
    entities_of_color_old = mesh_coloring->second.second;
  dolfin_assert(colors_old.size() == num_cells);
  dolfin_assert(!entities_of_color_old.empty());

  // Get coordinates
  const double* coordinates = mesh.geometry().coordinates;

  // New vertex indices, -1 if not yet renumbered
  std::vector<int> new_vertex_indices(num_vertices, -1);

  // Iterate over colors
  const uint num_colors = entities_of_color_old.size();
  uint connections_offset = 0;
  uint current_vertex = 0;
  for (uint color = 0; color < num_colors; ++color)
  {
    // Get the array of cell indices of current color
    const std::vector<uint>& colored_cells = entities_of_color_old[color];

    // Iterate over cells for current color
    for (uint i = 0; i < colored_cells.size(); i++)
    {
      // Current cell
      Cell cell(mesh, colored_cells[i]);

      // Get array of vertices for current cell
      const uint* cell_vertices = cell.entities(0);
      const uint num_vertices   = cell.num_entities(0);

      // Iterate over cell vertices
      for (uint j = 0; j < num_vertices; j++)
      {
        // Get vertex index
        const uint vertex_index = cell_vertices[j];

        // Renumber and copy coordinate data if vertex is not yet renumbered
        if (new_vertex_indices[vertex_index] == -1)
        {
          std::copy(coordinates + vertex_index*gdim,
                    coordinates + (vertex_index + 1)*gdim,
                    new_coordinates.begin() + current_vertex*gdim);
          new_vertex_indices[vertex_index] = current_vertex++;
        }

        // Renumber and copy connectivity data (must be done after vertex renumbering)
        const uint new_vertex_index = new_vertex_indices[vertex_index];
        dolfin_assert(new_vertex_index >= 0);
        new_connections[connections_offset++] = new_vertex_index;
      }
    }
  }

  // Check that all vertices have been renumbered
  for (uint i = 0; i < new_vertex_indices.size(); i++)
  {
    if (new_vertex_indices[i] == -1)
      dolfin_error("MeshRenumbering.cpp",
                   "compute renumbering of mesh",
                   "Some vertices were not renumbered");
  }
}
//-----------------------------------------------------------------------------
