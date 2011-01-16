// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2011.
//
// First added:  2010-11-27
// Last changed: 2011-01-16

#include <algorithm>
#include <vector>
#include <boost/scoped_array.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include "Cell.h"
#include "Mesh.h"
#include "MeshData.h"
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshRenumbering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshRenumbering::renumber_by_color(Mesh& mesh,
                                  boost::tuple<uint, uint, uint> coloring_type)
{
  info("Renumbering mesh by cell colors.");
  info(mesh);

  // Check that requested coloring is a cell coloring
  if (coloring_type.get<0>() != mesh.topology().dim())
    error("MeshRenumbering::renumber_by_color supports cell colorings only.");

  // Get coloring
  std::map<boost::tuple<uint, uint, uint>, std::pair<MeshFunction<uint>,
           std::vector<std::vector<uint> > > >::iterator mesh_coloring;
  mesh_coloring = mesh.data().coloring.find(coloring_type);

  // Check that requested coloring has been computed
  if (mesh_coloring == mesh.data().coloring.end())
    error("Requested mesh coloring has not been computed. Cannot renumber");

  // Get coloring data
  MeshFunction<uint>& colors = mesh_coloring->second.first;
  std::vector<std::vector<uint> >& entities_of_color = mesh_coloring->second.second;
  assert(colors.size() == mesh.num_cells());
  assert(entities_of_color.size() > 0);

  // Get mesh topology and geometry
  MeshTopology& topology = mesh.topology();
  MeshGeometry& geometry = mesh.geometry();

  // Issue warning if connectivity other than cell-vertex exists
  const uint D = topology.dim();
  for (uint d0 = 0; d0 <= D; d0++)
    for (uint d1 = 0; d1 <= D; d1++)
      if (!(d0 == D && d1 == 0) && topology(d0, d1).size() > 0)
        warning("Clearing connectivity data %d - %d.", d0, d1);

  // Clean connectivity since it may be invalid after renumbering
  mesh.clean();

  // Start timer
  Timer timer("Renumber mesh");

  // Get connectivity and coordinates
  MeshConnectivity& connectivity = topology(D, 0);
  uint* connections = connectivity.connections;
  double* coordinates = geometry.coordinates;

  // Allocate temporary arrays, used for copying data
  const uint connections_size = connectivity._size;
  const uint coordinates_size = geometry.size()*geometry.dim();
  boost::scoped_array<uint> new_connections(new uint[connections_size]);
  boost::scoped_array<double> new_coordinates(new double[coordinates_size]);

  // New vertex indices, -1 if not yet renumbered
  std::vector<int> new_vertex_indices(mesh.num_vertices());
  std::fill(new_vertex_indices.begin(), new_vertex_indices.end(), -1);

  // Iterate over colors
  const uint num_colors = entities_of_color.size();
  uint connections_offset = 0;
  uint current_vertex = 0;
  for (uint color = 0; color < num_colors; color++)
  {
    // Get the array of cell indices of current color
    std::vector<uint>& colored_cells = entities_of_color[color];

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
        const uint vertex_index = cell_vertices[j];

        // Renumber and copy coordinate data
        if (new_vertex_indices[vertex_index] == -1)
        {
          const uint d = mesh.geometry().dim();
          std::copy(coordinates + vertex_index*d,
                    coordinates + (vertex_index + 1)*d,
                    new_coordinates.get() + current_vertex*d);
          new_vertex_indices[vertex_index] = current_vertex++;
        }

        // Renumber and copy connectivity data (must be done after vertex renumbering)
        const uint new_vertex_index = new_vertex_indices[vertex_index];
        assert(new_vertex_index >= 0);
        new_connections[connections_offset++] = new_vertex_index;
      }
    }
  }

  // Check that all vertices have been renumbered
  for (uint i = 0; i < new_vertex_indices.size(); i++)
    if (new_vertex_indices[i] == -1)
      error("Failed to renumber mesh, not all vertices renumbered.");

  // Copy data
  std::copy(new_connections.get(), new_connections.get() + connections_size, connections);
  std::copy(new_coordinates.get(), new_coordinates.get() + coordinates_size, coordinates);

  // Update renumbering data
  uint current_cell = 0;
  for (uint color = 0; color < num_colors; color++)
  {
    // Get the array of cell indices of current color
    std::vector<uint>& colored_cells = entities_of_color[color];

    // Update cell color data
    for (uint i = 0; i < colored_cells.size(); i++)
    {
      colored_cells[i] = current_cell;
      colors[current_cell] = color;
      current_cell++;
    }
  }
}
//-----------------------------------------------------------------------------
