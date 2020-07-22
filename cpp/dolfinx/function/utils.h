// Copyright (C) 2013-2020 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Expression.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Mesh.h>

namespace dolfinx {

namespace mesh
{
class Mesh;
class Topology;
} // namespace mesh

namespace fem {
template <typename T>
class FormCoefficients;
}

namespace function {

template <typename T>
class Expression;

/// Pack expression coefficients ready for assembly
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
pack_coefficients(const function::Expression<T>& e)
{
  // Get expression coefficient offsets amd dofmaps
  const fem::FormCoefficients<T>& coefficients = e.coefficients();
  const std::vector<int>& offsets = coefficients.offsets();
  std::vector<const fem::DofMap*> dofmaps(coefficients.size());
  std::vector<Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>> v;
  for (int i = 0; i < coefficients.size(); ++i)
  {
    dofmaps[i] = coefficients.get(i)->function_space()->dofmap().get();
    v.emplace_back(coefficients.get(i)->x()->array());
  }

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = e.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells
      = mesh->topology().index_map(tdim)->size_local()
        + mesh->topology().index_map(tdim)->num_ghosts();

  // Copy data into coefficient array
  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c(
      num_cells, offsets.back());
  if (coefficients.size() > 0)
  {
    for (int cell = 0; cell < num_cells; ++cell)
    {
      for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
      {
        auto dofs = dofmaps[coeff]->cell_dofs(cell);
        const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& _v
            = v[coeff];
        for (Eigen::Index k = 0; k < dofs.size(); ++k)
          c(cell, k + offsets[coeff]) = _v[dofs[k]];
      }
    }
  }

  return c;
}

// NOTE: This is subject to change
/// Pack form constants ready for assembly
template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1> pack_constants(const function::Expression<T>& e)
{
  std::vector<T> constant_values;
  for (auto& constant : e.constants())
  {
    const std::vector<T>& array = constant.second->value;
    constant_values.insert(constant_values.end(), array.begin(), array.end());
  }

  return Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(
      constant_values.data(), constant_values.size(), 1);
}

} // namespace function
} // namespace dolfinx
