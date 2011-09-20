// Copyright (C) 2008 Ola Skavhaug
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
// Modified by Anders Logg, 2008-2009.
//
// First added:  2008-11-28
// Last changed: 2011-03-17

#include <dolfin/common/MPI.h>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "Mesh.h"
#include "MeshDomains.h"
#include "Vertex.h"
#include "LocalMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData() : num_global_vertices(0), num_global_cells(0),
                                 num_vertices_per_cell(0), gdim(0), tdim(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData(const Mesh& mesh) : num_global_vertices(0),
               num_global_cells(0), num_vertices_per_cell(0), gdim(0), tdim(0)
{
  // Extract data on main process and split among processes
  if (MPI::is_broadcaster())
  {
    extract_mesh_data(mesh);
    broadcast_mesh_data();
  }
  else
    receive_mesh_data();
}
//-----------------------------------------------------------------------------
LocalMeshData::~LocalMeshData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::string LocalMeshData::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false);
    s << std::endl;

    s << "  Vertex coordinates" << std::endl;
    s << "  ------------------" << std::endl;
    for (uint i = 0; i < vertex_coordinates.size(); i++)
    {
      s << "    " << i << ":";
      for (uint j = 0; j < vertex_coordinates[i].size(); j++)
        s << " " << vertex_coordinates[i][j];
      s << std::endl;
    }
    s << std::endl;

    s << "  Vertex indices" << std::endl;
    s << "  --------------" << std::endl;
    for (uint i = 0; i < vertex_coordinates.size(); i++)
      s << "    " << i << ": " << vertex_indices[i] << std::endl;
    s << std::endl;

    s << "  Cell vertces" << std::endl;
    s << "  ------------" << std::endl;
    for (uint i = 0; i < cell_vertices.size(); i++)
    {
      s << "    " << i << ":";
      for (uint j = 0; j < cell_vertices[i].size(); j++)
        s << " " << cell_vertices[i][j];
      s << std::endl;
    }
    s << std::endl;
  }
  else
  {
    s << "<LocalMeshData on process "
      << MPI::process_number() << " with "
      << vertex_coordinates.size() << " vertices (out of "
      << num_global_vertices << ") and "
      << cell_vertices.size() << " cells (out of "
      << num_global_cells << ")>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void LocalMeshData::clear()
{
  vertex_coordinates.clear();
  vertex_indices.clear();
  cell_vertices.clear();
  global_cell_indices.clear();
  num_global_vertices = 0;
  num_global_cells = 0;
  gdim = 0;
  tdim = 0;
  domain_data.clear();
}
//-----------------------------------------------------------------------------
void LocalMeshData::extract_mesh_data(const Mesh& mesh)
{
  if (!mesh.domains().is_empty())
    error("LocalMeshData::extract_mesh_data does not yet support marked domains.");

  // Clear old data
  clear();

  // Set scalar data
  gdim = mesh.geometry().dim();
  tdim = mesh.topology().dim();
  num_global_vertices = mesh.num_vertices();
  num_global_cells = mesh.num_cells();
  num_vertices_per_cell = mesh.type().num_entities(0);

  /// Get coordinates for all vertices stored on local processor
  vertex_coordinates.reserve(mesh.num_vertices());
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    std::vector<double> coordinates(gdim);
    for (uint i = 0; i < gdim; ++i)
      coordinates[i] = vertex->x()[i];
    vertex_coordinates.push_back(coordinates);
  }

  /// Get global vertex indices for all vertices stored on local processor
  vertex_indices.reserve(mesh.num_vertices());
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    vertex_indices.push_back(vertex->index());

  /// Get global vertex indices for all cells stored on local processor
  cell_vertices.reserve(mesh.num_cells());
  global_cell_indices.reserve(mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    global_cell_indices.push_back((*cell).index());
    std::vector<uint> vertices(cell->num_entities(0));
    for (uint i = 0; i < cell->num_entities(0); ++i)
      vertices[i] = cell->entities(0)[i];
    cell_vertices.push_back(vertices);
  }

  cout << "Number of global vertices: " << num_global_vertices << endl;
  cout << "Number of global cells: "    << num_global_cells << endl;
}
//-----------------------------------------------------------------------------
void LocalMeshData::broadcast_mesh_data()
{
  // Get number of processes
  const uint num_processes = MPI::num_processes();

  dolfin_debug("check");
  // Broadcast simple scalar data
  {
    std::vector<uint> values;
    values.push_back(gdim);
    values.push_back(tdim);
    values.push_back(num_global_vertices);
    values.push_back(num_global_cells);
    values.push_back(num_vertices_per_cell);
    MPI::broadcast(values);
  }

  dolfin_debug("check");
  /// Broadcast coordinates for vertices
  {
    std::vector<std::vector<double> > values(num_processes);
    for (uint p = 0; p < num_processes; p++)
    {
      std::pair<uint, uint> local_range = MPI::local_range(p, num_global_vertices);
      log(TRACE, "Sending %d vertices to process %d, range is (%d, %d)",
          local_range.second - local_range.first, p, local_range.first, local_range.second);
      for (uint i = local_range.first; i < local_range.second; i++)
      {
        for (uint j = 0; j < vertex_coordinates[i].size(); j++)
          values[p].push_back(vertex_coordinates[i][j]);
      }
    }
    MPI::scatter(values);
    unpack_vertex_coordinates(values[0]);
  }

  dolfin_debug("check");
  /// Broadcast global vertex indices
  {
    std::vector<std::vector<uint> > values(num_processes);
    for (uint p = 0; p < num_processes; p++)
    {
      std::pair<uint, uint> local_range = MPI::local_range(p, num_global_vertices);
      for (uint i = local_range.first; i < local_range.second; i++)
        values[p].push_back(vertex_indices[i]);
    }
    MPI::scatter(values);
    unpack_vertex_indices(values[0]);
  }

  dolfin_debug("check");
  /// Broadcast cell vertices
  {
    std::vector<std::vector<uint> > values(num_processes);
    for (uint p = 0; p < num_processes; p++)
    {
      std::pair<uint, uint> local_range = MPI::local_range(p, num_global_cells);
      log(TRACE, "Sending %d cells to process %d, range is (%d, %d)",
          local_range.second - local_range.first, p, local_range.first, local_range.second);
      for (uint i = local_range.first; i < local_range.second; i++)
      {
        values[p].push_back(global_cell_indices[i]);
        for (uint j = 0; j < cell_vertices[i].size(); j++)
          values[p].push_back(cell_vertices[i][j]);
      }
    }
    MPI::scatter(values);
    unpack_cell_vertices(values[0]);
  }
}
//-----------------------------------------------------------------------------
void LocalMeshData::receive_mesh_data()
{
  dolfin_debug("check");
  // Receive simple scalar data
  {
    std::vector<uint> values;
    MPI::broadcast(values);
    assert(values.size() == 5);
    gdim = values[0];
    tdim = values[1];
    num_global_vertices = values[2];
    num_global_cells = values[3];
    num_vertices_per_cell = values[4];
  }

  dolfin_debug("check");
  /// Receive coordinates for vertices
  {
    std::vector<std::vector<double> > values;
    MPI::scatter(values);
    unpack_vertex_coordinates(values[0]);
  }

  dolfin_debug("check");
  /// Receive global vertex indices
  {
    std::vector<std::vector<uint> > values;
    MPI::scatter(values);
    unpack_vertex_indices(values[0]);
  }

  dolfin_debug("check");
  /// Receive coordinates for vertices
  {
    std::vector<std::vector<uint> > values;
    MPI::scatter(values);
    unpack_cell_vertices(values[0]);
  }
}
//-----------------------------------------------------------------------------
void LocalMeshData::unpack_vertex_coordinates(const std::vector<double>& values)
{
  assert(values.size() % gdim == 0);
  vertex_coordinates.clear();
  const uint num_vertices = values.size() / gdim;
  uint k = 0;
  for (uint i = 0; i < num_vertices; i++)
  {
    std::vector<double> coordinates(gdim);
    for (uint j = 0; j < gdim; j++)
      coordinates[j] = values[k++];
    vertex_coordinates.push_back(coordinates);
  }

  log(TRACE, "Received %d vertex coordinates", vertex_coordinates.size());
}
//-----------------------------------------------------------------------------
void LocalMeshData::unpack_vertex_indices(const std::vector<uint>& values)
{
  assert(values.size() == vertex_coordinates.size());
  vertex_indices.clear();
  for (uint i = 0; i < values.size(); i++)
    vertex_indices.push_back(values[i]);

  log(TRACE, "Received %d vertex indices", vertex_coordinates.size());
}
//-----------------------------------------------------------------------------
void LocalMeshData::unpack_cell_vertices(const std::vector<uint>& values)
{
  assert(values.size() % (tdim + 2) == 0);
  cell_vertices.clear();
  global_cell_indices.clear();
  const uint num_cells = values.size() / (tdim + 2);
  uint k = 0;
  for (uint i = 0; i < num_cells; i++)
  {
    global_cell_indices.push_back(values[k++]);
    std::vector<uint> vertices(tdim + 1);
    for (uint j = 0; j < tdim + 1; j++)
      vertices[j] = values[k++];
    cell_vertices.push_back(vertices);
  }

  log(TRACE, "Received %d cell vertices", cell_vertices.size());
}
//-----------------------------------------------------------------------------
