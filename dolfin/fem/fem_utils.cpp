// Copyright (C) 2013, 2015, 2016 Johan Hake, Jan Blechta
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

#include <dolfin/common/ArrayView.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/la/GenericVector.h>

#include "fem_utils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<std::size_t> dolfin::dof_to_vertex_map(const FunctionSpace& space)
{
  // Get vertex_to_dof_map and invert it
  const std::vector<dolfin::la_index> vertex_map = vertex_to_dof_map(space);
  std::vector<std::size_t> return_map(vertex_map.size());
  for (std::size_t i = 0; i < vertex_map.size(); i++)
  {
    return_map[vertex_map[i]] = i;
  }
  return return_map;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index>
dolfin::vertex_to_dof_map(const FunctionSpace& space)
{
  // Get the mesh
  dolfin_assert(space.mesh());
  dolfin_assert(space.dofmap());
  const Mesh& mesh = *space.mesh();
  const GenericDofMap& dofmap = *space.dofmap();

  if (dofmap.is_view())
  {
    dolfin_error("fem_utils.cpp",
                 "tabulate vertex to dof map",
                 "Cannot tabulate vertex_to_dof_map for a subspace");
  }

  // Initialize vertex to cell connections
  const std::size_t top_dim = mesh.topology().dim();
  mesh.init(0, top_dim);

  // Num dofs per vertex
  const std::size_t dofs_per_vertex = dofmap.num_entity_dofs(0);
  const std::size_t vert_per_cell = mesh.topology()(top_dim, 0).size(0);
  if (vert_per_cell*dofs_per_vertex != dofmap.max_element_dofs())
  {
    dolfin_error("DofMap.cpp",
                 "tabulate dof to vertex map",
                 "Can only tabulate dofs on vertices");
  }

  // Allocate data for tabulating local to local map
  std::vector<std::size_t> local_to_local_map(dofs_per_vertex);

  // Create return data structure
  std::vector<dolfin::la_index>
    return_map(dofs_per_vertex*mesh.num_entities(0));

  // Iterate over vertices
  std::size_t local_vertex_ind = 0;
  for (VertexIterator vertex(mesh, "all"); !vertex.end(); ++vertex)
  {
    // Get the first cell connected to the vertex
    const Cell cell(mesh, vertex->entities(top_dim)[0]);

    // Find local vertex number
#ifdef DEBUG
    bool vertex_found = false;
#endif
    for (std::size_t i = 0; i < cell.num_entities(0); i++)
    {
      if (cell.entities(0)[i] == vertex->index())
      {
        local_vertex_ind = i;
#ifdef DEBUG
        vertex_found = true;
#endif
        break;
      }
    }
    dolfin_assert(vertex_found);

    // Get all cell dofs
    const ArrayView<const dolfin::la_index> cell_dofs
      = dofmap.cell_dofs(cell.index());

    // Tabulate local to local map of dofs on local vertex
    dofmap.tabulate_entity_dofs(local_to_local_map, 0,
				local_vertex_ind);

    // Fill local dofs for the vertex
    for (std::size_t local_dof = 0; local_dof < dofs_per_vertex; local_dof++)
    {
      const dolfin::la_index global_dof
        = cell_dofs[local_to_local_map[local_dof]];
      return_map[dofs_per_vertex*vertex->index() + local_dof] = global_dof;
    }
  }

  // Return the map
  return return_map;
}
//-----------------------------------------------------------------------------
void _check_coordinates(const MeshGeometry& geometry, const Function& position)
{
  dolfin_assert(position.function_space());
  dolfin_assert(position.function_space()->mesh());
  dolfin_assert(position.function_space()->dofmap());
  dolfin_assert(position.function_space()->element());
  dolfin_assert(position.function_space()->element()->ufc_element());

  if (position.function_space()->element()->ufc_element()->family()
          != std::string("Lagrange"))

  {
    dolfin_error("fem_utils.cpp",
                 "set/get mesh geometry coordinates from/to function",
                 "expecting 'Lagrange' finite element family rather than '%s'",
                 position.function_space()->element()->ufc_element()->family());
  }

  if (position.value_rank() != 1)
  {
    dolfin_error("fem_utils.cpp",
                 "set/get mesh geometry coordinates from/to function",
                 "function has incorrect value rank %d, need 1",
                 position.value_rank());
  }

  if (position.value_dimension(0) != geometry.dim())
  {
    dolfin_error("fem_utils.cpp",
                 "set/get mesh geometry coordinates from/to function",
                 "function value dimension %d and geometry dimension %d "
                 "do not match",
                 position.value_dimension(0), geometry.dim());
  }

  if (position.function_space()->element()->ufc_element()->degree()
          != geometry.degree())
  {
    dolfin_error("fem_utils.cpp",
                 "set/get mesh geometry coordinates from/to function",
                 "function degree %d and geometry degree %d do not match",
                 position.function_space()->element()->ufc_element()->degree(),
                 geometry.degree());
  }
}
//-----------------------------------------------------------------------------
// This helper function sets geometry from position (if setting) or stores
// geometry into position (otherwise)
void _get_set_coordinates(MeshGeometry& geometry, Function& position,
  const bool setting)
{
  auto& x = geometry.x();
  auto& v = *position.vector();
  const auto& dofmap = *position.function_space()->dofmap();
  const auto& mesh = *position.function_space()->mesh();
  const auto tdim = mesh.topology().dim();
  const auto gdim = mesh.geometry().dim();

  std::vector<std::size_t> num_local_entities(tdim+1);
  std::vector<std::size_t> coords_per_entity(tdim+1);
  std::vector<std::vector<std::vector<std::size_t>>> local_to_local(tdim+1);
  std::vector<std::vector<std::size_t>> offsets(tdim+1);

  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    // Get number local entities
    num_local_entities[dim] = mesh.type().num_entities(dim);

    // Get local-to-local mapping of dofs
    local_to_local[dim].resize(num_local_entities[dim]);
    for (std::size_t local_ind = 0; local_ind != num_local_entities[dim]; ++local_ind)
      dofmap.tabulate_entity_dofs(local_to_local[dim][local_ind], dim, local_ind);

    // Get entity offsets; could be retrieved directly from geometry
    coords_per_entity[dim] = geometry.num_entity_coordinates(dim);
    for (std::size_t coord_ind = 0; coord_ind != coords_per_entity[dim]; ++coord_ind)
    {
      const auto offset = geometry.get_entity_index(dim, coord_ind, 0);
      offsets[dim].push_back(offset);
    }
  }

  // Initialize needed connectivities
  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    if (coords_per_entity[dim] > 0)
      mesh.init(tdim, dim);
  }

  ArrayView<const la_index> cell_dofs;
  std::vector<double> values;
  const unsigned int* global_entities;
  std::size_t xi, vi;

  // Get/set cell-by-cell
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    // Get/prepare values and dofs on cell
    cell_dofs = dofmap.cell_dofs(c->index());
    values.resize(cell_dofs.size());
    if (setting)
      v.get_local(values.data(), cell_dofs.size(), cell_dofs.data());

    // Iterate over all entities on cell
    for (std::size_t dim = 0; dim <= tdim; ++dim)
    {
      // Get local-to-global entity mapping
      global_entities = c->entities(dim);

      for (std::size_t local_entity = 0;
           local_entity != num_local_entities[dim]; ++local_entity)
      {
        for (std::size_t local_dof = 0; local_dof != coords_per_entity[dim];
              ++local_dof)
        {
          for (std::size_t component = 0; component != gdim; ++component)
          {
            // Compute indices
            xi = gdim*(offsets[dim][local_dof] + global_entities[local_entity])
               + component;
            vi = local_to_local[dim][local_entity][gdim*local_dof + component];

            // Set one or other
            if (setting)
              x[xi] = values[vi];
            else
              values[vi] = x[xi];
          }
        }
      }
    }

    // Store cell contribution to dof vector (if getting)
    if (!setting)
      v.set_local(values.data(), cell_dofs.size(), cell_dofs.data());
  }

  if (!setting)
    v.apply("insert");
}
//-----------------------------------------------------------------------------
void dolfin::set_coordinates(MeshGeometry& geometry, const Function& position)
{
  _check_coordinates(geometry, position);
  _get_set_coordinates(geometry, const_cast<Function&>(position), true);
}
//-----------------------------------------------------------------------------
void dolfin::get_coordinates(Function& position, const MeshGeometry& geometry)
{
  _check_coordinates(geometry, position);
  _get_set_coordinates(const_cast<MeshGeometry&>(geometry), position, false);
}
//-----------------------------------------------------------------------------
Mesh dolfin::create_mesh(const Function& coordinates)
{
  dolfin_assert(coordinates.function_space());
  dolfin_assert(coordinates.function_space()->element());
  dolfin_assert(coordinates.function_space()->element()->ufc_element());

  // Fetch old mesh and create new mesh
  const Mesh& mesh0 = *(coordinates.function_space()->mesh());
  Mesh mesh1(mesh0.mpi_comm());

  // Assign all data except geometry
  mesh1._topology = mesh0._topology;
  mesh1._domains = mesh0._domains;
  mesh1._data = mesh0._data;
  if (mesh0._cell_type)
    mesh1._cell_type.reset(CellType::create(mesh0._cell_type->cell_type()));
  else
    mesh1._cell_type.reset();
  mesh1._ordered = mesh0._ordered;
  mesh1._cell_orientations = mesh0._cell_orientations;

  // Rename
  mesh1.rename(mesh0.name(), mesh0.label());

  // Call assignment operator for base class
  static_cast<Hierarchical<Mesh>>(mesh1) = mesh0;

  // Prepare a new geometry
  mesh1._geometry.init(mesh0._geometry.dim(),
    coordinates.function_space()->element()->ufc_element()->degree());
  std::vector<std::size_t> num_entities(mesh0._topology.dim() + 1);
  for (std::size_t dim = 0; dim <= mesh0._topology.dim(); ++dim)
    num_entities[dim] = mesh0._topology.size(dim);
  mesh1._geometry.init_entities(num_entities);

  // Assign coordinates
  set_coordinates(mesh1._geometry, coordinates);

  return mesh1;
}
