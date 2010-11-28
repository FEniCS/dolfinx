// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-27
// Last changed: 2010-11-28

#include <algorithm>
#include <boost/scoped_array.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/Timer.h>
#include "Cell.h"
#include "Mesh.h"
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshRenumbering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshRenumbering::renumber_by_color(Mesh& mesh)
{
  info(TRACE, "Renumbering mesh by cell colors.");

  // Get mesh topology and geometry
  MeshTopology& topology = mesh.topology();
  MeshGeometry& geometry = mesh.geometry();

  // Check that only cell-vertex connectivity exists
  const uint D = topology.dim();
  for (uint d0 = 0; d0 <= D; d0++)
  {
    for (uint d1 = 0; d1 <= D; d1++)
    {
      if (!(d0 == D && d1 == 0) && topology(d0, d1).size() > 0)
      {
        error("Unable to renumber mesh by colors. Only cell-vertex connectivity may exist prior to renumbering.");
      }
    }
  }

  // Start timer
  Timer timer("Renumber mesh");

  // Get connectivity and coordinates
  MeshConnectivity& connectivity = topology(D, 0);
  uint* connections = connectivity.connections;
  double* coordinates = geometry.coordinates;

  // Allocate temporary arrays, used for copying data
  boost::scoped_array<uint> new_connections(new uint[connectivity._size]);
  boost::scoped_array<double> new_coordinates(new double[geometry._size]);

  // New vertex indices, -1 if not yet renumbered
  std::vector<int> new_vertex_indices(mesh.num_vertices());
  std::fill(new_vertex_indices.begin(), new_vertex_indices.end(), -1);

  // Iterate over colors
  const uint num_colors = mesh.data().array("num colored cells")->size();
  for (uint color = 0; color < num_colors; color++)
  {
    // Get the array of cell indices of current color
    const std::vector<uint>* colored_cells = mesh.data().array("colored cells", color);
    if (!colored_cells)
      error("Unable to renumber mesh by colors. Mesh has not been colored.");

    // Iterate over cells for current color
    uint connections_offset = 0;
    uint current_vertex = 0;
    for (uint i = 0; i < colored_cells->size(); i++)
    {
      // Current cell
      Cell cell(mesh, (*colored_cells)[i]);

      // Get array of vertices for current cell
      const uint* cell_vertices = cell.entities(0);
      const uint num_vertices = cell.num_entities(0);

      // Iterate over cell vertices
      for (uint j = 0; j < num_vertices; j++)
      {
        const uint vertex_index = cell_vertices[j];

        // Renumber and copy coordinate data
        if (new_vertex_indices[vertex_index] == -1)
        {
          const uint d = mesh.geometry().dim();
          std::copy(coordinates + vertex_index * d,
                    coordinates + (vertex_index + 1) * d,
                    new_coordinates.get() + current_vertex * d);
          new_vertex_indices[vertex_index] = current_vertex++;
        }

        // Renumber and copy connectivity data (must be done after vertex renumbering)
        const uint new_vertex_index = new_vertex_indices[vertex_index];
        assert(new_vertex_index >= 0);
        new_connections[connections_offset++] = new_vertex_index;
      }
    }
  }

  // Copy data
  std::copy(new_connections.get(), new_connections.get() + connectivity._size, connections);
  std::copy(new_coordinates.get(), new_coordinates.get() + geometry._size, coordinates);
}
//-----------------------------------------------------------------------------
