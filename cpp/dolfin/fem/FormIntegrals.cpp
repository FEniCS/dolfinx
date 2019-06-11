// Copyright (C) 2018 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormIntegrals.h"
#include <cstdlib>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Facet.h>
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
const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                         const int*, const int*)>&
FormIntegrals::get_tabulate_tensor_function(FormIntegrals::Type type,
                                            unsigned int i) const
{
  int type_index = static_cast<int>(type);
  const std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals[type_index];

  if (i > integrals.size())
    throw std::runtime_error("Invalid integral index: " + std::to_string(i));

  return integrals[i].tabulate;
}
//-----------------------------------------------------------------------------
void FormIntegrals::register_tabulate_tensor(FormIntegrals::Type type, int i,
                                             void (*fn)(PetscScalar*,
                                                        const PetscScalar*,
                                                        const double*,
                                                        const int*, const int*))
{
  const int type_index = static_cast<int>(type);
  std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals[type_index];

  if (std::find_if(integrals.begin(), integrals.end(),
                   [&i](const struct FormIntegrals::Integral& val) {
                     return val.id == i;
                   })
      != integrals.end())
  {
    throw std::runtime_error("Integral with ID " + std::to_string(i)
                             + " already exists");
  }

  // Find insertion position
  int pos = std::distance(
      integrals.begin(),
      std::upper_bound(
          integrals.begin(), integrals.end(), i,
          [](const int& aid, const struct FormIntegrals::Integral& b) {
            return aid < b.id;
          }));

  // Create new Integral
  struct FormIntegrals::Integral new_integral
      = {fn, i, std::vector<std::int32_t>()};

  integrals.insert(integrals.begin() + pos, new_integral);
}
//-----------------------------------------------------------------------------
int FormIntegrals::num_integrals(FormIntegrals::Type type) const
{
  int type_index = static_cast<int>(type);
  return _integrals[type_index].size();
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
FormIntegrals::integral_domains(FormIntegrals::Type type, unsigned int i) const
{
  int type_index = static_cast<int>(type);
  if (i >= _integrals[type_index].size())
    throw std::runtime_error("Invalid integral:" + std::to_string(i));
  return _integrals[type_index][i].active_entities;
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
  if (type == Type::exterior_facet or type == Type::interior_facet)
    --tdim;
  else if (type == Type::vertex)
    tdim = 1;

  if (tdim != marker.dim())
  {
    throw std::runtime_error("Invalid MeshFunction dimension:"
                             + std::to_string(marker.dim()));
  }

  // Create a reverse map
  std::map<int, int> id_to_integral;
  for (unsigned int i = 0; i < integrals.size(); ++i)
  {
    if (integrals[i].id != -1)
    {
      integrals[i].active_entities.clear();
      id_to_integral[integrals[i].id] = i;
    }
  }

  for (unsigned int i = 0; i < marker.size(); ++i)
  {
    auto it = id_to_integral.find(marker[i]);
    if (it != id_to_integral.end())
      integrals[it->second].active_entities.push_back(i);
  }
}
//-----------------------------------------------------------------------------
void FormIntegrals::set_default_domains(const mesh::Mesh& mesh)
{
  int tdim = mesh.topology().dim();

  std::vector<struct FormIntegrals::Integral>& integrals
      = _integrals[static_cast<int>(FormIntegrals::Type::cell)];

  // If there is a default integral, define it on all cells
  if (integrals.size() > 0 and integrals[0].id == -1)
  {
    integrals[0].active_entities.resize(mesh.num_entities(tdim));
    std::iota(integrals[0].active_entities.begin(),
              integrals[0].active_entities.end(), 0);
  }

  integrals = _integrals[static_cast<int>(FormIntegrals::Type::exterior_facet)];
  if (integrals.size() > 0 and integrals[0].id == -1)
  {
    // If there is a default integral, define it only on surface facets
    integrals[0].active_entities.clear();
    for (const mesh::Facet& facet : mesh::MeshRange<mesh::Facet>(mesh))
    {
      if (facet.num_global_entities(tdim) == 1)
        integrals[0].active_entities.push_back(facet.index());
    }
  }

  integrals = _integrals[static_cast<int>(FormIntegrals::Type::interior_facet)];
  if (integrals.size() > 0 and integrals[0].id == -1)
  {
    // If there is a default integral, define it only on interior facets
    integrals[0].active_entities.clear();
    integrals[0].active_entities.reserve(mesh.num_entities(tdim - 1));
    for (const mesh::Facet& facet : mesh::MeshRange<mesh::Facet>(mesh))
    {
      if (facet.num_global_entities(tdim) != 1)
        integrals[0].active_entities.push_back(facet.index());
    }
  }
}
//-----------------------------------------------------------------------------
