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
// Modified by Anders Logg 2008-2011
//
// First added:  2008-11-28
// Last changed: 2012-11-24

#include <utility>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "Mesh.h"
#include "MeshDomains.h"
#include "Vertex.h"
#include "LocalMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData(const MPI_Comm mpi_comm) : _mpi_comm(mpi_comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData(const Mesh& mesh) : _mpi_comm(mesh.mpi_comm())
{
  Timer timer("Build LocalMeshData from local Mesh");

  // Extract data on main process and split among processes
  if (MPI::is_broadcaster(mesh.mpi_comm()))
  {
    extract_mesh_data(mesh);
    broadcast_mesh_data(mesh.mpi_comm());
  }
  else
    receive_mesh_data(mesh.mpi_comm());
}
//-----------------------------------------------------------------------------
LocalMeshData::~LocalMeshData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LocalMeshData::check() const
{
  dolfin_assert(geometry.num_global_vertices != -1);
  dolfin_assert(topology.num_global_cells != -1);
  dolfin_assert(topology.num_vertices_per_cell  != -1);
  dolfin_assert(geometry.dim  != -1);
  dolfin_assert(topology.dim  != -1);
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
    for (std::size_t i = 0; i < geometry.vertex_coordinates.size(); i++)
    {
      s << "    " << i << ":";
      for (std::size_t j = 0; j < geometry.vertex_coordinates[i].size(); j++)
        s << " " << geometry.vertex_coordinates[i][j];
      s << std::endl;
    }
    s << std::endl;

    s << "  Vertex indices" << std::endl;
    s << "  --------------" << std::endl;
    for (std::size_t i = 0; i < geometry.vertex_coordinates.size(); i++)
      s << "    " << i << ": " << geometry.vertex_indices[i] << std::endl;
    s << std::endl;

    s << "  Cell vertices" << std::endl;
    s << "  ------------" << std::endl;
    for (std::size_t i = 0; i < topology.cell_vertices.shape()[0]; i++)
    {
      s << "    " << i << ":";
      for (std::size_t j = 0; j < topology.cell_vertices.shape()[1]; j++)
        s << " " << topology.cell_vertices[i][j];
      s << std::endl;
    }
    s << std::endl;
  }
  else
  {
    s << "<LocalMeshData with "
      << geometry.vertex_coordinates.size() << " vertices (out of "
      << geometry.num_global_vertices << ") and "
      << topology.cell_vertices.shape()[0] << " cells (out of "
      << topology.num_global_cells << ")>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void LocalMeshData::clear()
{
  geometry.clear();
  topology.clear();
  domain_data.clear();
}
//-----------------------------------------------------------------------------
void LocalMeshData::extract_mesh_data(const Mesh& mesh)
{
  if (!mesh.domains().is_empty())
  {
    dolfin_error("LocalMeshData.cpp",
                 "extract local mesh data",
                 "Marked subdomains are not yet supported");
  }

  // Clear old data
  clear();

  // Set scalar data
  geometry.dim = mesh.geometry().dim();
  topology.dim = mesh.topology().dim();
  geometry.num_global_vertices = mesh.num_vertices();
  topology.num_global_cells = mesh.num_cells();
  topology.num_vertices_per_cell = mesh.type().num_entities(0);
  topology.cell_type = mesh.type().cell_type();

  // Get coordinates for all vertices stored on local processor
  geometry.vertex_coordinates.resize(boost::extents[mesh.num_vertices()][geometry.dim]);
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    const std::size_t index = vertex->index();
    std::copy(vertex->x(), vertex->x() + geometry.dim,
              geometry.vertex_coordinates[index].begin());
  }

  // Get global vertex indices for all vertices stored on local processor
  geometry.vertex_indices.reserve(mesh.num_vertices());
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
    geometry.vertex_indices.push_back(vertex->index());

  // Get global vertex indices for all cells stored on local processor
  topology.cell_vertices.resize(boost::extents[mesh.num_cells()][topology.num_vertices_per_cell]);
  topology.global_cell_indices.reserve(mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::size_t index = cell->index();
    topology.global_cell_indices.push_back(index);
    std::copy(cell->entities(0), cell->entities(0) + topology.num_vertices_per_cell,
              topology.cell_vertices[index].begin());
  }
}
//-----------------------------------------------------------------------------
void LocalMeshData::broadcast_mesh_data(const MPI_Comm mpi_comm)
{
  // Get number of processes
  const std::size_t num_processes = MPI::size(mpi_comm);

  // Broadcast simple int scalar data
  {
    std::vector<std::int64_t> values;
    values.push_back(geometry.dim);
    values.push_back(topology.dim);
    values.push_back(geometry.num_global_vertices);
    values.push_back(topology.num_global_cells);
    values.push_back(topology.num_vertices_per_cell);
    values.push_back(topology.cell_type);
    MPI::broadcast(mpi_comm, values);
  }

  // Broadcast coordinates for vertices
  {
    std::vector<std::vector<double>> send_values(num_processes);
    for (std::size_t p = 0; p < num_processes; p++)
    {
      const std::pair<std::size_t, std::size_t> local_range
        = MPI::local_range(mpi_comm, p, geometry.num_global_vertices);
      log(TRACE, "Sending %d vertices to process %d, range is (%d, %d)",
          local_range.second - local_range.first, p, local_range.first,
          local_range.second);

      send_values[p].reserve(geometry.dim*(local_range.second - local_range.first));
      for (std::size_t i = local_range.first; i < local_range.second; i++)
      {
        send_values[p].insert(send_values[p].end(),
                              geometry.vertex_coordinates[i].begin(),
                              geometry.vertex_coordinates[i].end());
      }
    }
    std::vector<double> values;
    MPI::scatter(mpi_comm, send_values, values);
    geometry.unpack_vertex_coordinates(values);
  }

  // Broadcast global vertex indices
  {
    std::vector<std::vector<std::int64_t>> send_values(num_processes);
    for (std::size_t p = 0; p < num_processes; p++)
    {
      const std::pair<std::size_t, std::size_t> local_range
        = MPI::local_range(mpi_comm, p, geometry.num_global_vertices);
      send_values[p].reserve(local_range.second - local_range.first);
      for (std::size_t i = local_range.first; i < local_range.second; i++)
        send_values[p].push_back(geometry.vertex_indices[i]);
    }
    MPI::scatter(mpi_comm, send_values, geometry.vertex_indices);
  }

  dolfin_debug("check");
  // Broadcast cell vertices
  {
    std::vector<std::vector<std::int64_t>> send_values(num_processes);
    for (std::size_t p = 0; p < num_processes; p++)
    {
      const std::pair<std::size_t, std::size_t> local_range
        = MPI::local_range(mpi_comm, p, topology.num_global_cells);
      log(TRACE, "Sending %d cells to process %d, range is (%d, %d)",
          local_range.second - local_range.first, p, local_range.first, local_range.second);
      const std::size_t range = local_range.second - local_range.first;
      send_values[p].reserve(range*(topology.num_vertices_per_cell + 1));
      for (std::size_t i = local_range.first; i < local_range.second; i++)
      {
        send_values[p].push_back(topology.global_cell_indices[i]);
        send_values[p].insert(send_values[p].end(),
                              topology.cell_vertices[i].begin(), topology.cell_vertices[i].end());
      }
    }
    std::vector<std::int64_t> values;
    MPI::scatter(mpi_comm, send_values, values);
    topology.unpack_cell_vertices(values);
  }
}
//-----------------------------------------------------------------------------
void LocalMeshData::receive_mesh_data(const MPI_Comm mpi_comm)
{
  dolfin_debug("check");

  // Receive simple scalar data
  {
    std::vector<std::int64_t> values;
    MPI::broadcast(mpi_comm, values);
    dolfin_assert(values.size() == 6);
    geometry.dim = values[0];
    topology.dim = values[1];
    geometry.num_global_vertices = values[2];
    topology.num_global_cells = values[3];
    topology.num_vertices_per_cell = values[4];
    topology.cell_type = (CellType::Type)values[5];
  }

  dolfin_debug("check");
  // Receive coordinates for vertices
  {
    std::vector<std::vector<double>> send_values;
    std::vector<double> values;
    MPI::scatter(mpi_comm, send_values, values);
    geometry.unpack_vertex_coordinates(values);
  }

  dolfin_debug("check");
  // Receive global vertex indices
  {
    std::vector<std::vector<std::int64_t>> send_values;
    MPI::scatter(mpi_comm, send_values, geometry.vertex_indices);
  }

  dolfin_debug("check");
  // Receive coordinates for vertices
  {
    std::vector<std::vector<std::int64_t>> send_values;
    std::vector<std::int64_t> values;
    MPI::scatter(mpi_comm, send_values, values);
    topology.unpack_cell_vertices(values);
  }
}
//-----------------------------------------------------------------------------
void
LocalMeshData::Geometry::unpack_vertex_coordinates(const std::vector<double>& values)
{
  dolfin_assert(values.size() % dim == 0);
  const std::size_t num_vertices = values.size()/dim;
  vertex_coordinates.resize(boost::extents[num_vertices][dim]);
  for (std::size_t i = 0; i < num_vertices; i++)
  {
    std::copy(values.begin() + i*dim, values.begin() + (i + 1)*dim,
              vertex_coordinates[i].begin());
  }

  log(TRACE, "Received %d vertex coordinates", vertex_coordinates.size());
}
//-----------------------------------------------------------------------------
void
LocalMeshData::Topology::unpack_cell_vertices(const std::vector<std::int64_t>& values)
{
  const std::size_t num_cells = values.size()/(num_vertices_per_cell + 1);
  dolfin_assert(values.size() % (num_vertices_per_cell + 1) == 0);
  cell_vertices.resize(boost::extents[num_cells][num_vertices_per_cell]);
  global_cell_indices.clear();
  std::size_t k = 0;
  for (std::size_t i = 0; i < num_cells; i++)
  {
    global_cell_indices.push_back(values[k++]);
    for (int j = 0; j < num_vertices_per_cell; j++)
      cell_vertices[i][j] = values[k++];
  }

  log(TRACE, "Received %d cell vertices", cell_vertices.size());
}
//-----------------------------------------------------------------------------
void LocalMeshData::reorder()
{
  const int dim0 = topology.cell_vertices.shape()[0];
  const int dim1 = topology.cell_vertices.shape()[1];
  dolfin_assert((int) topology.global_cell_indices.size() == dim0);

  // Build a vector of first vertex index for each cell in vertex_indices
  std::vector<std::pair<std::int64_t, std::int32_t>> keys(dim0);
  for (int i = 0; i < dim0; ++i)
    keys[i] = {*std::min_element(topology.cell_vertices[i].begin(), topology.cell_vertices[i].end()), i};

  // Sort
  std::sort(keys.begin(), keys.end());

  // Copy cell_vertices and cell local-to-global array
  boost::multi_array<std::int64_t, 2> _cell_vertices(boost::extents[dim0][dim1]);
  _cell_vertices = topology.cell_vertices;
  std::vector<std::int64_t> _global_cell_indices = topology.global_cell_indices;

  // Re-map data
  for (int i = 0; i < dim0; ++i)
  {
    auto key = keys[i];
    topology.global_cell_indices[i] = _global_cell_indices[key.second];
    for (int j = 0; j < dim1; ++j)
      topology.cell_vertices[i][j] = _cell_vertices[key.second][j];
  }
}
//-----------------------------------------------------------------------------
