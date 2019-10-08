// Copyright (C) 2018 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormIntegrals.h"
#include <cstdlib>
#include <dolfin/common/types.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FormIntegrals::FormIntegrals()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                         const double*, const int*, const int*)>&
FormIntegrals::get_tabulate_tensor_function(FormIntegrals::Type type,
                                            int i) const
{
  int type_index = static_cast<int>(type);
  const std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals.at(type_index);
  return integrals.at(i).tabulate;
}
//-----------------------------------------------------------------------------
void FormIntegrals::register_tabulate_tensor(
    FormIntegrals::Type type, int i,
    void (*fn)(PetscScalar*, const PetscScalar*, const PetscScalar*,
               const double*, const int*, const int*))
{
  const int type_index = static_cast<int>(type);
  std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals.at(type_index);

  // Find insertion point
  int pos = 0;
  for (const auto& q : integrals)
  {
    if (q.id == i)
    {
      throw std::runtime_error("Integral with ID " + std::to_string(i)
                               + " already exists");
    }
    else if (q.id > i)
      break;
    ++pos;
  }

  // Create new Integral and insert
  struct FormIntegrals::Integral new_integral
      = {fn, i, std::vector<std::int32_t>()};

  integrals.insert(integrals.begin() + pos, new_integral);
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_integrals(FormIntegrals::Type type) const
{
  return _integrals[static_cast<int>(type)].size();
}
//-----------------------------------------------------------------------------
std::vector<int> FormIntegrals::integral_ids(FormIntegrals::Type type) const
{
  std::vector<int> ids;
  int type_index = static_cast<int>(type);
  for (auto& integral : _integrals[type_index])
    ids.push_back(integral.id);

  return ids;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>&
FormIntegrals::integral_domains(FormIntegrals::Type type, int i) const
{
  int type_index = static_cast<int>(type);
  return _integrals[type_index].at(i).active_entities;
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_domains(FormIntegrals::Type type,
                                const mesh::MeshFunction<std::size_t>& marker)
{
  int type_index = static_cast<int>(type);
  std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals[type_index];

  if (integrals.size() == 0)
    return;

  std::shared_ptr<const mesh::Mesh> mesh = marker.mesh();
  int tdim = mesh->topology().dim();
  int dim = tdim;
  if (type == Type::exterior_facet or type == Type::interior_facet)
    dim = tdim - 1;
  else if (type == Type::vertex)
    dim = 0;

  if (dim != marker.dim())
  {
    throw std::runtime_error("Invalid MeshFunction dimension:"
                             + std::to_string(marker.dim()));
  }

  // Create a reverse map
  std::map<int, int> id_to_integral;
  for (std::size_t i = 0; i < integrals.size(); ++i)
  {
    if (integrals[i].id != -1)
    {
      integrals[i].active_entities.clear();
      id_to_integral[integrals[i].id] = i;
    }
  }

  // Get reference to mesh function data array
  const Eigen::Array<std::size_t, Eigen::Dynamic, 1>& mf_values
      = marker.values();
  const int num_entities = mesh->topology().ghost_offset(dim);

  if (type == Type::exterior_facet)
  {
    std::shared_ptr<const mesh::Connectivity> connectivity
        = mesh->topology().connectivity(tdim - 1, tdim);
    if (!connectivity)
    {
      throw std::runtime_error(
          "Facet-cell connectivity has not been computed.");
    }
    for (Eigen::Index i = 0; i < num_entities; ++i)
    {
      if ((int)connectivity->size_global(i) == 1)
      {
        auto it = id_to_integral.find(mf_values[i]);
        if (it != id_to_integral.end())
          integrals[it->second].active_entities.push_back(i);
      }
    }
  }
  else if (type == Type::interior_facet)
  {
    const int rank = MPI::rank(mesh->mpi_comm());
    const std::vector<std::int32_t>& cell_owners
        = mesh->topology().entity_owner(tdim);
    const std::int32_t cell_ghost_offset = mesh->topology().ghost_offset(tdim);
    std::shared_ptr<const mesh::Connectivity> connectivity
        = mesh->topology().connectivity(tdim - 1, tdim);
    if (!connectivity)
    {
      throw std::runtime_error(
          "Facet-cell connectivity has not been computed.");
    }
    for (Eigen::Index i = 0; i < num_entities; ++i)
    {
      if ((int)connectivity->size(i) == 2)
      {
        // Get connected cells and check if they are ghost or not
        const std::int32_t* c = connectivity->connections(i);
        const int owner0 = c[0] >= cell_ghost_offset
                               ? cell_owners[c[0] - cell_ghost_offset]
                               : rank;
        const int owner1 = c[1] >= cell_ghost_offset
                               ? cell_owners[c[1] - cell_ghost_offset]
                               : rank;
        if ((owner0 == rank and owner1 == rank)
            or (owner0 == rank and owner1 > rank)
            or (owner1 == rank and owner0 > rank))
        {
          auto it = id_to_integral.find(mf_values[i]);
          if (it != id_to_integral.end())
            integrals[it->second].active_entities.push_back(i);
        }
      }
    }
  }
  else
  {
    // For cell and vertex integrals use all markers (but not on ghost entities)
    for (Eigen::Index i = 0; i < num_entities; ++i)
    {
      auto it = id_to_integral.find(mf_values[i]);
      if (it != id_to_integral.end())
        integrals[it->second].active_entities.push_back(i);
    }
  }
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_default_domains(const mesh::Mesh& mesh)
{
  const int tdim = mesh.topology().dim();

  std::vector<struct FormIntegrals::Integral>& cell_integrals
      = _integrals[static_cast<int>(FormIntegrals::Type::cell)];

  // If there is a default integral, define it on all cells (excluding
  // ghost cells)
  if (cell_integrals.size() > 0 and cell_integrals[0].id == -1)
  {
    const int num_regular_cells = mesh.topology().ghost_offset(tdim);
    cell_integrals[0].active_entities.resize(num_regular_cells);
    std::iota(cell_integrals[0].active_entities.begin(),
              cell_integrals[0].active_entities.end(), 0);
  }

  std::vector<struct FormIntegrals::Integral>& exf_integrals
      = _integrals[static_cast<int>(FormIntegrals::Type::exterior_facet)];
  if (exf_integrals.size() > 0 and exf_integrals[0].id == -1)
  {
    // If there is a default integral, define it only on surface facets
    exf_integrals[0].active_entities.clear();
    std::shared_ptr<const mesh::Connectivity> connectivity
        = mesh.topology().connectivity(tdim - 1, tdim);
    if (!connectivity)
    {
      throw std::runtime_error(
          "Facet-cell connectivity has not been computed.");
    }
    for (const mesh::MeshEntity& facet :
         mesh::MeshRange(mesh, tdim - 1, mesh::MeshRangeType::REGULAR))
    {
      if (connectivity->size_global(facet.index()) == 1)
        exf_integrals[0].active_entities.push_back(facet.index());
    }
  }

  std::vector<struct FormIntegrals::Integral>& inf_integrals
      = _integrals[static_cast<int>(FormIntegrals::Type::interior_facet)];
  if (inf_integrals.size() > 0 and inf_integrals[0].id == -1)
  {
    // If there is a default integral, define it only on interior facets
    inf_integrals[0].active_entities.clear();
    inf_integrals[0].active_entities.reserve(mesh.num_entities(tdim - 1));
    const int rank = MPI::rank(mesh.mpi_comm());
    if (MPI::size(mesh.mpi_comm()) > 1)
    {
      // Get owner (MPI ranks) of ghost cells
      const std::vector<std::int32_t>& cell_owners
          = mesh.topology().entity_owner(tdim);
      const std::int32_t ghost_offset = mesh.topology().ghost_offset(tdim);

      std::shared_ptr<const mesh::Connectivity> connectivity
          = mesh.topology().connectivity(tdim - 1, tdim);
      if (!connectivity)
      {
        throw std::runtime_error(
            "Facet-cell connectivity has not been computed.");
      }

      for (const mesh::MeshEntity& facet :
           mesh::MeshRange(mesh, tdim - 1, mesh::MeshRangeType::ALL))
      {
        if (connectivity->size(facet.index()) == 2)
        {
          const std::int32_t* c = facet.entities(tdim);
          const int owner0
              = c[0] >= ghost_offset ? cell_owners[c[0] - ghost_offset] : rank;
          const int owner1
              = c[1] >= ghost_offset ? cell_owners[c[1] - ghost_offset] : rank;
          if ((owner0 == rank and owner1 == rank)
              or (owner0 == rank and owner1 > rank)
              or (owner1 == rank and owner0 > rank))
            inf_integrals[0].active_entities.push_back(facet.index());
        }
      }
    }
    else
    {
      std::shared_ptr<const mesh::Connectivity> connectivity
          = mesh.topology().connectivity(tdim - 1, tdim);
      if (!connectivity)
      {
        throw std::runtime_error(
            "Facet-cell connectivity has not been computed.");
      }

      for (const mesh::MeshEntity& facet :
           mesh::MeshRange(mesh, tdim - 1, mesh::MeshRangeType::REGULAR))
      {
        if (connectivity->size_global(facet.index()) != 1)
          inf_integrals[0].active_entities.push_back(facet.index());
      }
    }
  }
}
//-----------------------------------------------------------------------------
