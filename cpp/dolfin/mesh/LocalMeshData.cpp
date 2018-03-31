// Copyright (C) 2008 Ola Skavhaug
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "LocalMeshData.h"
#include "Cell.h"
#include "Mesh.h"
#include "MeshIterator.h"
#include "Vertex.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <utility>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData(const MPI_Comm mpi_comm) : _mpi_comm(mpi_comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData(const Mesh& mesh) : _mpi_comm(mesh.mpi_comm())
{
  common::Timer timer("Build LocalMeshData from local Mesh");

  const std::size_t gdim = mesh.geometry().dim();
  geometry.num_global_vertices = mesh.num_entities_global(0);
  // Whatever process 0 says, copy to others
  MPI::broadcast(mpi_comm(), geometry.num_global_vertices);
  geometry.dim = gdim;
  geometry.vertex_coordinates.resize(
      boost::extents[mesh.num_entities(0)][gdim]);
  // std::copy(mesh.geometry().x().begin(), mesh.geometry().x().end(),
  //           geometry.vertex_coordinates.data());
  std::copy(mesh.geometry().points().data(),
            mesh.geometry().points().data() + mesh.geometry().points().size(),
            geometry.vertex_coordinates.data());
  geometry.vertex_indices.resize(mesh.num_entities(0));
  std::copy(mesh.topology().global_indices(0).begin(),
            mesh.topology().global_indices(0).end(),
            geometry.vertex_indices.begin());

  const std::size_t tdim = mesh.topology().dim();
  topology.num_global_cells = mesh.num_entities_global(tdim);
  // Whatever process 0 says, copy to others
  MPI::broadcast(mpi_comm(), topology.num_global_cells);
  topology.dim = tdim;
  topology.ordered = mesh.ordered();
  topology.num_vertices_per_cell = mesh.type().num_vertices();
  topology.cell_type = mesh.type().cell_type();
  topology.cell_vertices.resize(
      boost::extents[mesh.num_entities(tdim)][topology.num_vertices_per_cell]);
  topology.global_cell_indices.resize(mesh.num_entities(tdim));
  std::copy(mesh.topology().global_indices(tdim).begin(),
            mesh.topology().global_indices(tdim).end(),
            topology.global_cell_indices.begin());

  for (auto& c : MeshRange<Cell>(mesh))
  {
    const std::size_t i = c.index();
    const std::int32_t* v = c.entities(0);
    for (int j = 0; j != topology.num_vertices_per_cell; ++j)
      topology.cell_vertices[i][j] = geometry.vertex_indices[v[j]];
  }
}
//-----------------------------------------------------------------------------
void LocalMeshData::check() const
{
  dolfin_assert(geometry.num_global_vertices != -1);
  dolfin_assert(topology.num_global_cells != -1);
  dolfin_assert(topology.num_vertices_per_cell != -1);
  dolfin_assert(geometry.dim != -1);
  dolfin_assert(topology.dim != -1);
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
    s << "<LocalMeshData with " << geometry.vertex_coordinates.size()
      << " vertices (out of " << geometry.num_global_vertices << ") and "
      << topology.cell_vertices.shape()[0] << " cells (out of "
      << topology.num_global_cells << ")>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
