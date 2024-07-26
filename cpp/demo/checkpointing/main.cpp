// ```text
// Copyright (C) 2024 Abdullah Mujahid
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Checkpointing
//

#include <adios2.h>
#include <dolfinx.h>
#include <dolfinx/io/ADIOS2_utils.h>
#include <dolfinx/io/checkpointing.h>
#include <mpi.h>

using namespace dolfinx;
using namespace dolfinx::io;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // Create mesh
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<float>>(mesh::create_rectangle<float>(
      MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {4, 4},
      mesh::CellType::quadrilateral, part));

  // Create cell meshtags
  int dim = geometry.dim();
  topology->create_entities(dim);
  const std::shared_ptr<const dolfinx::common::IndexMap> topo_imap
      = topology->index_map(dim);

  std::uint32_t num_entities = topo_imap->size_local();

  auto cmap = mesh->geometry().cmap();
  auto geom_layout = cmap.create_dof_layout();
  std::uint32_t num_dofs_per_entity = geom_layout.num_entity_closure_dofs(dim);

  std::vector<int32_t> entities_array(num_entities * num_dofs_per_entity);
  std::vector<int32_t> entities_offsets(num_entities + 1);
  std::uint64_t offset = topo_imap->local_range()[0];
  std::vector<double> values(num_entities);

  for (int i = 0; i < values.size(); ++i)
  {
    values[i] = (double)(i + offset);
  }

  for (int i = 0; i < values.size() + 1; ++i)
    entities_offsets[i] = entities->offsets()[i];

  for (int i = 0; i < (int)(num_entities * num_dofs_per_entity); ++i)
    entities_array[i] = entities->array()[i];

  graph::AdjacencyList<std::int32_t> entities_local(entities_array,
                                                    entities_offsets);

  auto meshtags = std::make_shared<mesh::MeshTags<U>>(
      mesh::create_meshtags<U>(topology, dim, entities_local, values));

  auto writer = ADIOS2Engine(mesh->comm(), "mesh.bp", "mesh-write", "BP5",
                             adios2::Mode::Write);

  io::checkpointing::write(writer, mesh);
  io::checkpointing::write(writer, meshtags);

  MPI_Finalize();
  return 0;
}
