// Copyright (C) 2008 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2009-08-06
//
// Modified by Anders Logg, 2008-2009.

#include <dolfin/log/log.h>
#include <dolfin/main/MPI.h>
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
  error("This should not be called");
  dolfin_debug("check");

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
void LocalMeshData::clear()
{
  vertex_coordinates.clear();
  vertex_indices.clear();
  cell_vertices.clear();
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
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    std::vector<uint> vertices(cell->num_entities(0));
    for (uint i = 0; i < cell->num_entities(0); ++i)
    {
      vertices[i] = cell->entities(0)[i];
    }
    cell_vertices.push_back(vertices);
  }
}
//-----------------------------------------------------------------------------
void LocalMeshData::broadcast_mesh_data()
{
  // Get number of processes
  const uint num_processes = MPI::num_processes();

  // Broadcast simple scalar data
  {
    std::vector<uint> values;
    values.clear();
    values.push_back(gdim);
    values.push_back(tdim);
    values.push_back(num_global_vertices);
    values.push_back(num_global_cells);
    MPI::scatter(values);
  }

  /// Broadcast coordinates for vertices
  {
    std::vector<std::vector<double> > values(num_processes);
    for (uint p = 0; p < num_processes; p++)
    {
      std::pair<uint, uint> local_range = MPI::local_range(p, num_global_vertices);
      for (uint i = local_range.first; i < local_range.second; i++)
      {
        for (uint j = 0; j < vertex_coordinates[i].size(); j++)
          values[p].push_back(vertex_coordinates[i][j]);
      }
    }
    MPI::scatter(values);
  }

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
  }

  /// Broadcast cell vertices
  {
    std::vector<std::vector<uint> > values(num_processes);
    for (uint p = 0; p < num_processes; p++)
    {
      std::pair<uint, uint> local_range = MPI::local_range(p, num_global_cells);
      for (uint i = local_range.first; i < local_range.second; i++)
      {
        for (uint j = 0; j < cell_vertices[i].size(); j++)
          values[p].push_back(cell_vertices[i][j]);
      }
    }
    MPI::scatter(values);
  }
}
//-----------------------------------------------------------------------------
void LocalMeshData::receive_mesh_data()
{
  // Receive simple scalar data
  {
    std::vector<uint> values;
    MPI::scatter(values);
    assert(values.size() == 4);
    gdim = values[0];
    tdim = values[1];
    num_global_vertices = values[2];
    num_global_cells = values[3];
  }

  /// Receive coordinates for vertices
  {
    std::vector<std::vector<double> > values;
    MPI::scatter(values);
    assert(values[0].size() % gdim == 0);
    vertex_coordinates.clear();
    const uint num_vertices = values[0].size() / gdim;
    uint k = 0;
    for (uint i = 0; i < num_vertices; i++)
    {
      std::vector<double> coordinates(gdim);
      for (uint j = 0; j < gdim; j++)
        coordinates[j] = values[0][k++];
      vertex_coordinates.push_back(coordinates);
    }
  }

  /// Receive global vertex indices
  {
    std::vector<std::vector<uint> > values;
    MPI::scatter(values);
    assert(values[0].size() == vertex_coordinates.size());
    vertex_indices.clear();
    for (uint i = 0; i < values[0].size(); i++)
      vertex_indices.push_back(values[0][i]);
  }

  /// Receive coordinates for vertices
  {
    std::vector<std::vector<uint> > values;
    MPI::scatter(values);
    assert(values[0].size() % (tdim + 1) == 0);
    cell_vertices.clear();
    const uint num_cells = values[0].size() / (tdim + 1);
    uint k = 0;
    for (uint i = 0; i < num_cells; i++)
    {
      std::vector<uint> vertices(tdim + 1);
      for (uint j = 0; j < tdim + 1; j++)
        vertices[j] = values[0][k++];
      cell_vertices.push_back(vertices);
    }
  }
}
//-----------------------------------------------------------------------------
