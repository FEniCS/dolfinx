// Copyright (C) 2025 Jørgen S. Dokken and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <mpi.h>

namespace dolfinx::fem
{

template <typename T, std::floating_point U>
class MPC
{
public:
  /// @brief A Multipoint Constraint
  /// @param V FunctionSpace
  /// @param constrained_dofs_local List of local constrained dofs
  /// @param reference_dofs_global List of global reference dofs with
  /// weights for each local constrained dof
  /// @note u_constrained = sum(u_ref * coeff_ref)
  /// @note If the FunctionSpace V has a block size, then the dofs in
  /// constrained_dofs_local and reference_dofs_global must be the fully
  /// expanded dof indices.
  /// @todo Add a constant term to the constraint, i.e. u_constrained =
  /// sum(u_ref * coeff_ref) + constant

  MPC(const FunctionSpace<U>& V,
      const std::vector<std::int32_t>& constrained_dofs_local,
      const std::vector<std::vector<std::pair<T, std::int64_t>>>&
          reference_dofs_global)
  {
    if (constrained_dofs_local.size() != reference_dofs_global.size())
      throw std::runtime_error(
          "Incompatible lists of constrained and reference dofs");

    std::shared_ptr<const fem::DofMap> dm = V.dofmap();
    int bs = dm->bs();
    int index_map_bs = dm->index_map_bs();

    // Get unique list of global dofs
    std::vector<std::int64_t> gl_dofs;
    for (std::size_t i = 0; i < reference_dofs_global.size(); ++i)
    {
      for (auto r : reference_dofs_global[i])
        gl_dofs.push_back(r.second);
    }
    std::sort(gl_dofs.begin(), gl_dofs.end());
    gl_dofs.erase(std::unique(gl_dofs.begin(), gl_dofs.end()), gl_dofs.end());

    // Copy existing ghosts
    std::vector<std::int64_t> ghost_dofs(dm->index_map->ghosts().begin(),
                                         dm->index_map->ghosts().end());
    std::vector<std::int32_t> ghost_owners(dm->index_map->owners().begin(),
                                           dm->index_map->owners().end());

    // Skip in serial
    int size = dolfinx::MPI::size(V.mesh()->comm());
    if (size > 1)
    {
      int rank = dolfinx::MPI::rank(V.mesh()->comm());
      // Compute owner of global reference dofs
      // Should be a better way to do this

      std::int64_t sendbuf = dm->index_map->local_range()[0];
      std::vector<std::int64_t> recvbuf(size + 1);
      MPI_Allgather(&sendbuf, 1, MPI_INT64_T, recvbuf.data(), 1, MPI_INT64_T,
                    V.mesh()->comm());
      recvbuf.back() = dm->index_map->size_global();

      std::vector<int> gl_owner;
      for (auto g : gl_dofs)
      {
        auto it = std::upper_bound(recvbuf.begin(), recvbuf.end(),
                                   g / index_map_bs);
        gl_owner.push_back(std::distance(recvbuf.begin(), it) - 1);
      }
      // If not already included, add new global dofs to ghosts
      for (std::size_t i = 0; i < gl_dofs.size(); ++i)
      {
        if (gl_owner[i] == rank)
          continue;
        auto it = std::find(ghost_dofs.begin(), ghost_dofs.end(),
                            gl_dofs[i] / index_map_bs);
        if (it == ghost_dofs.end())
        {
          ghost_dofs.push_back(gl_dofs[i] / index_map_bs);
          ghost_owners.push_back(gl_owner[i]);
        }
      }
    }

    // New DofMap and FunctionSpace with extra ghost dofs
    std::shared_ptr<const common::IndexMap> index_map
        = std::make_shared<const common::IndexMap>(dm->index_map->comm(),
                                                   dm->index_map->size_local(),
                                                   ghost_dofs, ghost_owners);
    std::vector<std::int32_t> cell_dofs(
        dm->map().data_handle(), dm->map().data_handle() + dm->map().size());

    _V = std::make_shared<const FunctionSpace<U>>(
        V.mesh(), V.element(),
        std::make_shared<const DofMap>(dm->element_dof_layout(), index_map,
                                       index_map_bs, cell_dofs, bs));

    // Compute offsets for flattened arrays of reference dofs and weights
    std::vector<std::int32_t> count(
        index_map_bs
            * (dm->index_map->size_local() + dm->index_map->num_ghosts()),
        0);
    for (std::size_t i = 0; i < constrained_dofs_local.size(); ++i)
      count[constrained_dofs_local[i]] += reference_dofs_global[i].size();
    std::vector<std::int32_t> dof_to_ref(count.size() + 1, 0);
    std::partial_sum(count.begin(), count.end(), std::next(dof_to_ref.begin()));

    // Flatten reference dofs and weight arrays, in correct order
    std::vector<std::int64_t> ref_dofs_tmp(dof_to_ref.back());
    std::vector<std::int64_t> ref_dofs_component(dof_to_ref.back());
    std::vector<T> ref_coeffs_flat(dof_to_ref.back());
    std::vector<std::pair<std::int32_t, T>> constraints_flat(dof_to_ref.back());

    for (std::size_t i = 0; i < constrained_dofs_local.size(); ++i)
    {
      std::int32_t index = dof_to_ref[constrained_dofs_local[i]];
      const auto& refs_i = reference_dofs_global[i];
      for (std::size_t j = 0; j < refs_i.size(); ++j)
      {
        ref_coeffs_flat[index + j] = refs_i[j].first;
        // Remove and store the component, reapply after converting to local
        // index
        spdlog::info(
            "ref_dofs_tmp[{}] = {} / {}, ref_dofs_component[{}] = {} % {}",
            index + j, refs_i[j].second, index_map_bs, index + j,
            refs_i[j].second, index_map_bs);

        ref_dofs_tmp[index + j] = refs_i[j].second / index_map_bs;
        ref_dofs_component[index + j] = refs_i[j].second % index_map_bs;
      }
    }

    // Convert reference dofs to local indexing
    std::vector<std::int32_t> ref_dofs_flat(ref_dofs_tmp.size());
    _V->dofmap()->index_map->global_to_local(ref_dofs_tmp, ref_dofs_flat);

    for (std::size_t i = 0; i < ref_dofs_flat.size(); ++i)
      constraints_flat[i]
          = {ref_dofs_flat[i] * index_map_bs + ref_dofs_component[i],
             ref_coeffs_flat[i]};

    _constraints
        = std::make_unique<graph::AdjacencyList<std::pair<std::int32_t, T>>>(
            constraints_flat, dof_to_ref);
  }

  /// @brief Get modified FunctionSpace containing reference dofs as ghosts
  /// @return The FunctionSpace associated with the MPC, including extra ghost
  /// dofs
  std::shared_ptr<const FunctionSpace<U>> V() const { return _V; }

  /// @brief Find cells which contain constrained dofs
  /// @return List of cells containing constrained dofs
  std::vector<std::int32_t> cells() const
  {
    // TODO: work on mixed-topology meshes.
    auto cell_dofs = _V->dofmap()->map();
    int bs = _V->dofmap()->bs();
    std::vector<std::int32_t> marked_cells;
    for (std::size_t i = 0; i < cell_dofs.extent(0); ++i)
    {
      for (std::size_t j = 0; j < cell_dofs.extent(1); ++j)
      {
        int index = cell_dofs(i, j);
        for (int k = 0; k < bs; ++k)
        {
          if (_constraints->num_links(index * bs + k) > 0)
          {
            marked_cells.push_back(i);
            break;
          }
        }
      }
    }
    marked_cells.erase(std::unique(marked_cells.begin(), marked_cells.end()),
                       marked_cells.end());
    return marked_cells;
  }

  /// @brief Return the list of constraints for each local dof (if any).
  /// For each local dof, the list contains pairs of reference dof index and
  /// coefficient.
  /// @note The dof indices are expanded to include the block size.
  /// @note The reference dofs are local indices in the modified FunctionSpace.
  const graph::AdjacencyList<std::pair<std::int32_t, T>>& constraints() const
  {
    return *_constraints;
  }

private:
  // Modified FunctionSpace with additional ghost dofs
  std::shared_ptr<const FunctionSpace<U>> _V;

  // List of constraints for each local DoF (if any).
  // For each local dof, the list contains pairs of reference dof index and
  // coefficient.
  std::unique_ptr<graph::AdjacencyList<std::pair<std::int32_t, T>>>
      _constraints;

  // Constants for the MPC, if any. For each local constrained dof.
  std::unique_ptr<graph::AdjacencyList<T>> _constants;
};

/// @brief Assemble a billinear form with MPC constraints into a `MatrixCSR`.
/// @tparam T Scalar type
/// @tparam U Geometry scalar type
/// @param mpc MultiPoint Constraint
/// @param A Matrix to assemble into
/// @param a Form to assemble
/// @param bcs Dirichlet boundary conditions
/// @note Matrix A must have appropriate sparsity set beforehand
template <typename T, std::floating_point U, int BS = 1>
void assemble_mpc(
    const MPC<T, U>& mpc, la::MatrixCSR<T>& A, const Form<T, U>& a,
    const std::vector<std::reference_wrapper<const DirichletBC<T, U>>>& bcs)
{
  if (mpc.V()->dofmap()->bs() == BS)
  {
    spdlog::info("Assemble MPC with bs={}", BS);
    auto mat_add = A.template mat_add_values<BS, BS>();
    assemble_matrix_mpc(mpc, mat_add, a, bcs);
  }
  else if constexpr (BS < 10)
  {
    assemble_mpc<T, U, BS + 1>(mpc, A, a, bcs);
  }
  else
    throw std::runtime_error("Block size not supported");
}

/// @brief Add to a sparsity pattern for a form with multipoint constraints
/// @tparam T Scalar type
/// @tparam U Mesh geometry type
/// @param pattern Sparsity pattern to build
/// @param form Form for which to build sparsity pattern
/// @param mpc Multipoint constraint
/// @note The pattern is not finalised, i.e. the caller is responsible
/// for calling SparsityPattern::assemble.
template <dolfinx::scalar T, std::floating_point U>
void build_sparsity_pattern_mpc(la::SparsityPattern& pattern,
                                const Form<T, U>& form, const MPC<T, U>& mpc)
{
  if (form.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot add to sparsity pattern. Form is not a bilinear.");
  }

  if (form.function_spaces()[0] != mpc.V()
      or form.function_spaces()[1] != mpc.V())
  {
    throw std::runtime_error(
        "Cannot add to sparsity pattern. Form function spaces do not match "
        "MPC function space.");
  }

  // Insert extra connectivity for cells containing constrained dofs
  // NB - only works if row and column function spaces are the same, which is
  // the case for now.
  int bs = mpc.V()->dofmap()->bs();
  std::vector<std::int32_t> dofs0, dofs1;
  for (std::int32_t cell : mpc.cells())
  {
    auto cell_dofs = mpc.V()->dofmap()->cell_dofs(cell);
    dofs0.clear();
    for (std::int32_t d : cell_dofs)
      for (int k = 0; k < bs; ++k)
        dofs0.push_back(d * bs + k);
    dofs1.clear();
    for (std::int32_t d : dofs0)
    {
      if (mpc.constraints().num_links(d) == 0)
        dofs1.push_back(d);
      else
      {
        for (auto [ref_dof, coeff] : mpc.constraints().links(d))
          dofs1.push_back(ref_dof);
      }
    }
    std::sort(dofs1.begin(), dofs1.end());
    dofs1.erase(std::unique(dofs1.begin(), dofs1.end()), dofs1.end());
    for (std::int32_t& d : dofs1)
      d /= bs;
    pattern.insert(dofs1, dofs1);
  }

  // Insert extra connectivity for reference dofs in the sparsity pattern
  auto constraints = mpc.constraints();
  std::vector<std::int32_t> ref_dofs;
  for (std::size_t dof = 0; dof < mpc.V()->dofmap()->index_map->size_local();
       ++dof)
  {
    for (int k = 0; k < bs; ++k)
    {
      if (constraints.num_links(dof * bs + k) > 0)
      {
        auto c = constraints.links(dof * bs + k);
        ref_dofs.resize(c.size());
        for (std::size_t i = 0; i < c.size(); ++i)
          ref_dofs[i] = c[i].first / bs;
        ref_dofs.push_back(dof);
        pattern.insert(ref_dofs, ref_dofs);
      }
    }
  }
}

} // namespace dolfinx::fem
