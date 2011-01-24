// Copyright (C) 2008 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2010-04-05
//
// Modified by Anders Logg, 2008-2009.

#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include "Mesh.h"
#include "Vertex.h"
#include "Cell.h"
#include "LocalMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData()
  : num_global_vertices(0), num_global_cells(0),
    gdim(0), tdim(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData(const Mesh& mesh)
  : num_global_vertices(0), num_global_cells(0),
    gdim(0), tdim(0)
{
  // Extract data on main process and split among processes
  if (MPI::is_broadcaster())
  {
    extract_mesh_data(mesh);
    broadcast_mesh_data();
  }
  else
  {
    receive_mesh_data();
  }
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
}
//-----------------------------------------------------------------------------
void LocalMeshData::extract_mesh_data(const Mesh& mesh)
{
  // Clear old data
  clear();

  // Set scalar data
  gdim = mesh.geometry().dim();
  tdim = mesh.topology().dim();
  num_global_vertices = mesh.num_vertices();
  num_global_cells = mesh.num_cells();

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
    {
      vertices[i] = cell->entities(0)[i];
    }
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
    std::vector<std::vector<uint> > values(num_processes);
    for (uint p = 0; p < num_processes; p++)
    {
      values[p].push_back(gdim);
      values[p].push_back(tdim);
      values[p].push_back(num_global_vertices);
      values[p].push_back(num_global_cells);
    }
    MPI::scatter(values);
  }

  dolfin_debug("check");
  /// Broadcast coordinates for vertices
  {
    std::vector<std::vector<double> > values(num_processes);
    for (uint p = 0; p < num_processes; p++)
    {
      std::pair<uint, uint> local_range = MPI::local_range(p, num_global_vertices);
      info(TRACE, "Sending %d vertices to process %d, range is (%d, %d)",
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
      info(TRACE, "Sending %d cells to process %d, range is (%d, %d)",
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
    std::vector<std::vector<uint> > values;
    MPI::scatter(values);
    assert(values[0].size() == 4);
    gdim = values[0][0];
    tdim = values[0][1];
    num_global_vertices = values[0][2];
    num_global_cells = values[0][3];
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

  info(TRACE, "Received %d vertex coordinates", vertex_coordinates.size());
}
//-----------------------------------------------------------------------------
void LocalMeshData::unpack_vertex_indices(const std::vector<uint>& values)
{
  assert(values.size() == vertex_coordinates.size());
  vertex_indices.clear();
  for (uint i = 0; i < values.size(); i++)
    vertex_indices.push_back(values[i]);

  info(TRACE, "Received %d vertex indices", vertex_coordinates.size());
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

  info(TRACE, "Received %d cell vertices", cell_vertices.size());
}
//-----------------------------------------------------------------------------
