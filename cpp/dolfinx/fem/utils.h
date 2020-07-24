// Copyright (C) 2013-2020 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace function
{
template <typename T>
class Constant;
template <typename T>
class Function;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Mesh;
class Topology;
} // namespace mesh

namespace fem
{

/// Extract test (0) and trial (1) function spaces pairs for each
/// bilinear form for a rectangular array of forms
///
/// @param[in] a A rectangular block on bilinear forms
/// @return Rectangular array of the same shape as @p a with a pair of
///   function spaces in each array entry. If a form is null, then the
///   returned function space pair is (null, null).
template <typename T>
Eigen::Array<std::array<std::shared_ptr<const function::FunctionSpace>, 2>,
             Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
extract_function_spaces(
    const Eigen::Ref<const Eigen::Array<const fem::Form<T>*, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>& a)
{
  Eigen::Array<std::array<std::shared_ptr<const function::FunctionSpace>, 2>,
               Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      spaces(a.rows(), a.cols());
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j)
      if (a(i, j))
        spaces(i, j) = {a(i, j)->function_space(0), a(i, j)->function_space(1)};
  return spaces;
}

/// Extract FunctionSpaces for (0) rows blocks and (1) columns blocks
/// from a rectangular array of bilinear forms. The test space must be
/// the same for each row and the trial spaces must be the same for each
/// column. Raises an exception if there is an inconsistency. e.g. if
/// each form in row i does not have the same test space then an
/// exception is raised.
///
/// @param[in] V Vector function spaces for (0) each row block and (1)
/// each column block
std::array<std::vector<std::shared_ptr<const function::FunctionSpace>>, 2>
common_function_spaces(
    const Eigen ::Ref<const Eigen::Array<
        std::array<std::shared_ptr<const function::FunctionSpace>, 2>,
        Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& V);

/// Create a sparsity pattern for a given form. The pattern is not
/// finalised, i.e. the caller is responsible for calling
/// SparsityPattern::assemble.
/// @param[in] a A bilinear form
/// @return The corresponding sparsity pattern
template <typename T>
la::SparsityPattern create_sparsity_pattern(const Form<T>& a)
{
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }

  // Get dof maps and mesh
  std::array dofmaps{a.function_space(0)->dofmap().get(),
                     a.function_space(1)->dofmap().get()};
  std::shared_ptr mesh = a.mesh();
  assert(mesh);

  const std::set<IntegralType> types = a.integrals().types();
  if (types.find(IntegralType::interior_facet) != types.end()
      or types.find(IntegralType::exterior_facet) != types.end())
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    const int tdim = mesh->topology().dim();
    mesh->topology_mutable().create_entities(tdim - 1);
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  }

  return create_sparsity_pattern(mesh->topology(), dofmaps, types);
}

/// Create a sparsity pattern for a given form. The pattern is not
/// finalised, i.e. the caller is responsible for calling
/// SparsityPattern::assemble.
la::SparsityPattern
create_sparsity_pattern(const mesh::Topology& topology,
                        const std::array<const DofMap*, 2>& dofmaps,
                        const std::set<IntegralType>& integrals);

/// Create an ElementDofLayout from a ufc_dofmap
ElementDofLayout create_element_dof_layout(const ufc_dofmap& dofmap,
                                           const mesh::CellType cell_type,
                                           const std::vector<int>& parent_map
                                           = {});

/// Create dof map on mesh from a ufc_dofmap
/// @param[in] comm MPI communicator
/// @param[in] dofmap The ufc_dofmap
/// @param[in] topology The mesh topology
DofMap create_dofmap(MPI_Comm comm, const ufc_dofmap& dofmap,
                     mesh::Topology& topology);

/// Create a CoordinateElement from ufc
/// @param[in] ufc_cmap UFC coordinate mapping
/// @return A DOLFINX coordinate map
fem::CoordinateElement
create_coordinate_map(const ufc_coordinate_mapping& ufc_cmap);

/// Create a CoordinateElement from ufc
/// @param[in] fptr Function Pointer to a ufc_function_coordinate_map
///   function
/// @return A DOLFINX coordinate map
fem::CoordinateElement create_coordinate_map(ufc_coordinate_mapping* (*fptr)());

/// Create FunctionSpace from UFC
/// @param[in] fptr Function Pointer to a ufc_function_space_create
///   function
/// @param[in] function_name Name of a function whose function space to
///   create. Function name is the name of Python variable for
///   ufl.Coefficient, ufl.TrialFunction or ufl.TestFunction as defined
///   in the UFL file.
/// @param[in] mesh Mesh
/// @return The created FunctionSpace
std::shared_ptr<function::FunctionSpace>
create_functionspace(ufc_function_space* (*fptr)(const char*),
                     const std::string function_name,
                     std::shared_ptr<mesh::Mesh> mesh);

// NOTE: This is subject to change
/// Pack form coefficients ready for assembly
template <typename T>
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
pack_coefficients(const fem::Form<T>& form)
{
  // Get form coefficient offsets amd dofmaps
  const fem::FormCoefficients<T>& coefficients = form.coefficients();
  const std::vector<int>& offsets = coefficients.offsets();
  std::vector<const fem::DofMap*> dofmaps(coefficients.size());
  std::vector<std::reference_wrapper<const Eigen::Matrix<T, Eigen::Dynamic, 1>>>
      v;
  for (int i = 0; i < coefficients.size(); ++i)
  {
    dofmaps[i] = coefficients.get(i)->function_space()->dofmap().get();
    v.emplace_back(coefficients.get(i)->x()->array());
  }

  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = form.mesh();
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
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& _v = v[coeff];
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
Eigen::Array<T, Eigen::Dynamic, 1> pack_constants(const fem::Form<T>& form)
{
  std::vector<T> constant_values;
  for (auto& constant : form.constants())
  {
    const std::vector<T>& array = constant.second->value;
    constant_values.insert(constant_values.end(), array.begin(), array.end());
  }

  return Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(
      constant_values.data(), constant_values.size(), 1);
}

} // namespace fem
} // namespace dolfinx
