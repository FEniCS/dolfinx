
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
