// Copyright (C) 2008 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-12-17
//
// Modified by Anders Logg, 2008.

#include <dolfin/log/log.h>
#include "LocalMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LocalMeshData::LocalMeshData()
  : num_global_vertices(0), num_global_cells(0),
    num_processes(0), process_number(0),
    gdim(0), tdim(0), cell_type(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LocalMeshData::~LocalMeshData()
{
  delete cell_type;
}
//-----------------------------------------------------------------------------
void LocalMeshData::clear()
{
  vertex_coordinates.clear();
  vertex_indices.clear();
  cell_vertices.clear();
}
//-----------------------------------------------------------------------------
dolfin::uint LocalMeshData::initial_vertex_location(uint vertex_index) const
{
  const uint num_vertices_per_process = num_global_vertices / num_processes;
  const uint remainder = num_global_vertices % num_processes;
  const uint breakpoint = remainder*(num_vertices_per_process + 1);
  if (vertex_index  < breakpoint)
    return vertex_index / (num_vertices_per_process + 1);
  return (vertex_index - breakpoint) / num_vertices_per_process + remainder;
}
//-----------------------------------------------------------------------------
dolfin::uint LocalMeshData::initial_cell_location(uint cell_index) const
{
  const uint num_cells_per_process = num_global_cells / num_processes;
  const uint remainder = num_global_cells % num_processes;
  if (cell_index < remainder)
    return cell_index / (num_cells_per_process + 1);
  return cell_index / num_cells_per_process;
}
//-----------------------------------------------------------------------------
dolfin::uint LocalMeshData::local_vertex_number(uint global_vertex_number) const
{
  uint start;
  uint stop;
  initial_vertex_range(start, stop);
  assert(global_vertex_number >= start);
  assert(global_vertex_number <= stop);
  return global_vertex_number - start;
}
//-----------------------------------------------------------------------------
void LocalMeshData::initial_vertex_range(uint& start, uint& stop) const
{
  // Compute number of vertices per process and remainder
  const uint n = num_global_vertices / num_processes;
  const uint r = num_global_vertices % num_processes;

  if (process_number < r)
  {
    start = process_number*(n + 1);
    stop = start + n;
  }
  else
  {
    start = process_number*n + r;
    stop = start + n - 1;
  }
}
//-----------------------------------------------------------------------------
void LocalMeshData::initial_cell_range(uint& start, uint& stop) const
{
  // Compute number of cells per process and remainder
  const uint n = num_global_cells / num_processes;
  const uint r = num_global_cells % num_processes;
  if (process_number < r)
  {
    start = process_number*(n + 1);
    stop = start + n;
  }
  else
  {
    start = process_number*n + r;
    stop = start + n - 1;
  }
}
//-----------------------------------------------------------------------------
