// Copyright (C) 2024 Abdullah Mujahid, Jørgen S. Dokken, Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "ADIOS2_utils.h"
#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

namespace
{
/// @brief Write a particular attribute incrementally.
/// For example, to write a new meshtag, fetch the name attribute if it exists
/// and append the name, otherwise create the attribute.
/// @tparam T ADIOS2 supported type
/// @param io ADIOS2 IO object
/// @param name Name of the attribute
/// @param value Value of the attribute to write
/// @param var_name Variable to which this attribute is associated with
/// @return Return the IO attribute
template <class T>
adios2::Attribute<T> define_attr(adios2::IO& io, const std::string& name,
                                 T& value, std::string var_name = "")
{
  bool modifiable = true;
  if (adios2::Attribute<T> attr = io.InquireAttribute<T>(name); attr)
  {
    std::vector<T> data = attr.Data();
    data.push_back(value);
    return io.DefineAttribute<T>(name, data.data(), data.size(), var_name, "/",
                                 modifiable);
  }
  else
  {
    std::vector<T> data{value};
    return io.DefineAttribute<T>(name, data.data(), data.size(), var_name, "/",
                                 modifiable);
  }
}

} // namespace

namespace dolfinx::io::native
{

/// @brief Write mesh to a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] mesh Mesh of type float or double to write to the file
template <std::floating_point T>
void write_mesh(adios2::IO& io, adios2::Engine& engine,
                const dolfinx::mesh::Mesh<T>& mesh);

/// @brief Read mesh from a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] comm comm
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return mesh reconstructed from the data
template <std::floating_point T>
dolfinx::mesh::Mesh<T> read_mesh(adios2::IO& io, adios2::Engine& engine,
                                 MPI_Comm comm = MPI_COMM_WORLD,
                                 dolfinx::mesh::GhostMode ghost_mode
                                 = dolfinx::mesh::GhostMode::shared_facet);

/// @brief Write meshtags to a file.
///
/// @tparam U float or double
/// @tparam T ADIOS2 supported type
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] mesh Mesh of type float or double to write to the file
/// @param[in] meshtags MeshTags to write to the file
template <std::floating_point U, typename T>
void write_meshtags(adios2::IO& io, adios2::Engine& engine,
                    dolfinx::mesh::Mesh<U>& mesh,
                    dolfinx::mesh::MeshTags<T>& meshtags)
{
  std::string name = meshtags.name;
  std::uint32_t dim = meshtags.dim();

  spdlog::info("Writing meshtags : {} for entities with dimension: {}", name,
               dim);

  {
    adios2::Attribute<std::string> attr_names
        = define_attr<std::string>(io, "meshtags_names", name);
    adios2::Attribute<std::uint32_t> attr_dims
        = define_attr<std::uint32_t>(io, "meshtags_dims", dim);
  }

  //
  std::uint32_t num_dofs_per_entity;
  {
    const fem::CoordinateElement<U>& cmap = mesh.geometry().cmap();
    fem::ElementDofLayout geom_layout = cmap.create_dof_layout();
    num_dofs_per_entity = geom_layout.num_entity_closure_dofs(dim);
  }

  std::uint64_t num_tag_entities_global, offset;

  // NOTE: For correctness of MPI_ calls, the following should match the
  // uint64_t type
  std::uint64_t num_tag_entities_local;
  std::vector<std::int64_t> array;
  {
    std::uint32_t num_entities_local;
    {
      std::shared_ptr<const mesh::Topology> topology = mesh.topology();
      assert(topology);

      num_entities_local = topology->index_map(dim)->size_local();
    }

    std::span<const std::int32_t> tag_entities = meshtags.indices();
    assert(std::ranges::is_sorted(tag_entities));

    std::uint32_t num_tag_entities = tag_entities.size();

    num_tag_entities_local
        = std::upper_bound(tag_entities.begin(), tag_entities.end(),
                           num_entities_local)
          - tag_entities.begin();

    // Compute the global offset for owned tagged entities
    offset = (std::uint64_t)0;
    MPI_Exscan(&num_tag_entities_local, &offset, 1, MPI_UINT64_T, MPI_SUM,
               mesh.comm());

    // Compute the global size of tagged entities
    num_tag_entities_global = (std::uint64_t)0;
    MPI_Allreduce(&num_tag_entities_local, &num_tag_entities_global, 1,
                  MPI_UINT64_T, MPI_SUM, mesh.comm());

    std::vector<std::int32_t> entities_to_geometry = mesh::entities_to_geometry(
        mesh, dim, tag_entities.subspan(0, num_tag_entities_local), false);

    array.resize(entities_to_geometry.size());

    std::iota(array.begin(), array.end(), 0);

    std::shared_ptr<const common::IndexMap> imap = mesh.geometry().index_map();
    imap->local_to_global(entities_to_geometry, array);
  }

  std::span<const T> values = meshtags.values();
  const std::span<const T> local_values(values.begin(), num_tag_entities_local);

  adios2::Variable<std::int64_t> var_topology = io.DefineVariable<std::int64_t>(
      name + "_topology", {num_tag_entities_global, num_dofs_per_entity},
      {offset, 0}, {num_tag_entities_local, num_dofs_per_entity},
      adios2::ConstantDims);

  adios2::Variable<T> var_values = io.DefineVariable<T>(
      name + "_values", {num_tag_entities_global}, {offset},
      {num_tag_entities_local}, adios2::ConstantDims);

  engine.BeginStep();
  engine.Put(var_topology, array.data());
  engine.Put(var_values, local_values.data());
  engine.EndStep();
}

} // namespace dolfinx::io::native

namespace dolfinx::io::impl_native
{
/// @brief Find offset and size.
///
/// @param[in] rank MPI rank
/// @param[in] N size of data to distribute
/// @param[in] size MPI size
/// @return start and count
std::pair<std::uint64_t, std::uint64_t> get_counters(int rank, std::uint64_t N,
                                                     int size);

/// @brief Read geometry data
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] dim The geometric dimension (`0 < dim <= 3`).
/// @param[in] num_nodes_global size of the global array of nodes
/// @param[in] rank MPI rank
/// @param[in] size MPI size
/// @return The point coordinates of row-major storage and
/// itsshape `(num_nodes_local, dim)`
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
read_geometry_data(adios2::IO& io, adios2::Engine& engine, int dim,
                   std::uint64_t num_nodes_global, int rank, int size);

/// @brief Read topology array
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] num_cells_global global number of cells
/// @param[in] rank MPI rank
/// @param[in] size MPI size
/// @return The cell-to-node connectivity in a flattened array
std::vector<int64_t> read_topology_data(adios2::IO& io, adios2::Engine& engine,
                                        std::uint64_t num_cells_global,
                                        int rank, int size);

/// @brief Read mesh from a file.
///
/// @param[in] io ADIOS2 IO
/// @param[in] engine ADIOS2 Engine
/// @param[in] comm comm
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return mesh reconstructed from the data
std::variant<dolfinx::mesh::Mesh<float>, dolfinx::mesh::Mesh<double>>
read_mesh_variant(adios2::IO& io, adios2::Engine& engine,
                  MPI_Comm comm = MPI_COMM_WORLD,
                  dolfinx::mesh::GhostMode ghost_mode
                  = dolfinx::mesh::GhostMode::shared_facet);

} // namespace dolfinx::io::impl_native

#endif
