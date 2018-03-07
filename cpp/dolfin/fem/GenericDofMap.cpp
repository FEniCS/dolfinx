
#include "GenericDofMap.h"
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshIterator.h>
#include <vector>

using namespace dolfin;
using namespace dolfin::fem;

std::vector<dolfin::la_index_t> GenericDofMap::dofs(const mesh::Mesh& mesh,
                                                    std::size_t dim) const
{
  // Check number of dofs per entity (on each cell)
  const std::size_t num_dofs_per_entity = num_entity_dofs(dim);

  // Return empty vector if not dofs on requested entity
  if (num_dofs_per_entity == 0)
    return std::vector<dolfin::la_index_t>();

  // Vector to hold list of dofs
  std::vector<dolfin::la_index_t> dof_list(mesh.num_entities(dim)
                                           * num_dofs_per_entity);

  // Iterate over cells
  std::vector<std::size_t> entity_dofs_local;
  for (auto& c : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Get local-to-global dofmap for cell
    const auto cell_dof_list = cell_dofs(c.index());

    // Loop over all entities of dimension dim
    unsigned int local_index = 0;
    for (auto& e : mesh::EntityRange<mesh::MeshEntity>(c, dim))
    {
      // Tabulate cell-wise index of all dofs on entity
      tabulate_entity_dofs(entity_dofs_local, dim, local_index);

      // Get dof index and add to list
      for (std::size_t i = 0; i < entity_dofs_local.size(); ++i)
      {
        const std::size_t entity_dof_local = entity_dofs_local[i];
        const dolfin::la_index_t dof_index = cell_dof_list[entity_dof_local];
        dolfin_assert(e.index() * num_dofs_per_entity + i < dof_list.size());
        dof_list[e.index() * num_dofs_per_entity + i] = dof_index;
      }

      ++local_index;
    }
  }

  return dof_list;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t>
GenericDofMap::entity_dofs(const mesh::Mesh& mesh, std::size_t entity_dim,
                           const std::vector<std::size_t>& entity_indices) const
{
  // Get some dimensions
  const std::size_t top_dim = mesh.topology().dim();
  const std::size_t dofs_per_entity = num_entity_dofs(entity_dim);

  // Initialize entity to cell connections
  mesh.init(entity_dim, top_dim);

  // Allocate the the array to return
  const std::size_t num_marked_entities = entity_indices.size();
  std::vector<dolfin::la_index_t> entity_to_dofs(num_marked_entities
                                                 * dofs_per_entity);

  // Allocate data for tabulating local to local map
  std::vector<std::size_t> local_to_local_map(dofs_per_entity);

  // Iterate over entities
  std::size_t local_entity_ind = 0;
  for (std::size_t i = 0; i < num_marked_entities; ++i)
  {
    mesh::MeshEntity entity(mesh, entity_dim, entity_indices[i]);

    // Get the first cell connected to the entity
    const mesh::Cell cell(mesh, entity.entities(top_dim)[0]);

    // Find local entity number
    for (std::size_t local_i = 0; local_i < cell.num_entities(entity_dim);
         ++local_i)
    {
      if (cell.entities(entity_dim)[local_i] == entity.index())
      {
        local_entity_ind = local_i;
        break;
      }
    }

    // Get all cell dofs
    const auto cell_dof_list = cell_dofs(cell.index());

    // Tabulate local to local map of dofs on local entity
    tabulate_entity_dofs(local_to_local_map, entity_dim, local_entity_ind);

    // Fill local dofs for the entity
    for (std::size_t local_dof = 0; local_dof < dofs_per_entity; ++local_dof)
    {
      // Map dofs
      const dolfin::la_index_t global_dof
          = cell_dof_list[local_to_local_map[local_dof]];
      entity_to_dofs[dofs_per_entity * i + local_dof] = global_dof;
    }
  }
  return entity_to_dofs;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::la_index_t>
GenericDofMap::entity_dofs(const mesh::Mesh& mesh, std::size_t entity_dim) const
{
  // Get some dimensions
  const std::size_t top_dim = mesh.topology().dim();
  const std::size_t dofs_per_entity = num_entity_dofs(entity_dim);
  const std::size_t num_mesh_entities = mesh.num_entities(entity_dim);

  // Initialize entity to cell connections
  mesh.init(entity_dim, top_dim);

  // Allocate the the array to return
  std::vector<dolfin::la_index_t> entity_to_dofs(num_mesh_entities
                                                 * dofs_per_entity);

  // Allocate data for tabulating local to local map
  std::vector<std::size_t> local_to_local_map(dofs_per_entity);

  // Iterate over entities
  std::size_t local_entity_ind = 0;
  for (auto& entity : mesh::MeshRange<mesh::MeshEntity>(mesh, entity_dim))
  {
    // Get the first cell connected to the entity
    const mesh::Cell cell(mesh, entity.entities(top_dim)[0]);

    // Find local entity number
    for (std::size_t local_i = 0; local_i < cell.num_entities(entity_dim);
         ++local_i)
    {
      if (cell.entities(entity_dim)[local_i] == entity.index())
      {
        local_entity_ind = local_i;
        break;
      }
    }

    // Get all cell dofs
    const auto cell_dof_list = cell_dofs(cell.index());

    // Tabulate local to local map of dofs on local entity
    tabulate_entity_dofs(local_to_local_map, entity_dim, local_entity_ind);

    // Fill local dofs for the entity
    for (std::size_t local_dof = 0; local_dof < dofs_per_entity; ++local_dof)
    {
      // Map dofs
      const dolfin::la_index_t global_dof
          = cell_dof_list[local_to_local_map[local_dof]];
      entity_to_dofs[dofs_per_entity * entity.index() + local_dof] = global_dof;
    }
  }
  return entity_to_dofs;
}
//-----------------------------------------------------------------------------
