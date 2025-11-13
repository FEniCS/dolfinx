// Copyright (C) 2025 Jørgen S. Dokken and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <concepts>
#include <dolfinx/fem/FunctionSpace.h>
#include <mpi.h>

#pragma once

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

  MPC(const FunctionSpace<U>& V,
      const std::vector<std::int32_t>& constrained_dofs_local,
      const std::vector<std::vector<std::pair<T, std::int64_t>>>&
          reference_dofs_global)
  {
    if (constrained_dofs_local.size() != reference_dofs_global.size())
      throw std::runtime_error(
          "Incompatible lists of constrained and reference dofs");

    std::shared_ptr<const fem::DofMap> dm = V.dofmap();

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
        auto it = std::upper_bound(recvbuf.begin(), recvbuf.end(), g);
        gl_owner.push_back(std::distance(recvbuf.begin(), it) - 1);
      }
      // If not already included, add new global dofs to ghosts
      for (std::size_t i = 0; i < gl_dofs.size(); ++i)
      {
        if (gl_owner[i] == rank)
          continue;
        auto it = std::find(ghost_dofs.begin(), ghost_dofs.end(), gl_dofs[i]);
        if (it == ghost_dofs.end())
        {
          ghost_dofs.push_back(gl_dofs[i]);
          ghost_owners.push_back(gl_owner[i]);
        }
      }
    }

    // New DofMap and FunctionSpace with extra ghost dofs
    int bs = dm->bs();
    int index_map_bs = dm->index_map_bs();
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
        dm->index_map->size_local() + dm->index_map->num_ghosts(), 0);
    for (std::size_t i = 0; i < constrained_dofs_local.size(); ++i)
      count[constrained_dofs_local[i]] += reference_dofs_global[i].size();
    dof_to_ref.resize(count.size() + 1, 0);
    std::partial_sum(count.begin(), count.end(), std::next(dof_to_ref.begin()));

    // Flatten reference dofs and weight arrays, in correct order
    std::vector<std::int64_t> ref_dofs_tmp(dof_to_ref.back());
    ref_coeffs_flat.resize(dof_to_ref.back());
    for (std::size_t i = 0; i < constrained_dofs_local.size(); ++i)
    {
      std::int32_t index = dof_to_ref[constrained_dofs_local[i]];
      const auto& refs_i = reference_dofs_global[i];
      for (std::size_t j = 0; j < refs_i.size(); ++j)
      {
        ref_coeffs_flat[index + j] = refs_i[j].first;
        ref_dofs_tmp[index + j] = refs_i[j].second;
      }
    }

    // Convert reference dofs to local indexing
    ref_dofs_flat.resize(ref_dofs_tmp.size());
    _V->dofmap()->index_map->global_to_local(ref_dofs_tmp, ref_dofs_flat);
  }

  /// @brief Get modified FunctionSpace containing reference dofs as ghosts
  /// @return The FunctionSpace associated with the MPC, including extra ghost
  /// dofs
  std::shared_ptr<const FunctionSpace<U>> V() { return _V; }

  /// @brief Find cells which contain constrained dofs
  /// @return List of cells containing constrained dofs
  std::vector<std::int32_t> cells() const
  {
    auto cell_dofs = _V->dofmap()->map();
    std::vector<std::int32_t> marked_cells;
    for (int i = 0; i < cell_dofs.extent(0); ++i)
    {
      for (int j = 0; j < cell_dofs.extent(1); ++j)
      {
        int index = cell_dofs(i, j);
        if (dof_to_ref[index] != dof_to_ref[index + 1])
          marked_cells.push_back(i);
      }
    }
    return marked_cells;
  }

  /// @brief Return constraint on a given dof, if any
  std::pair<std::vector<std::int32_t>, std::vector<T>>
  constraint(std::int32_t dof)
  {
    std::vector<std::int32_t> ref_dofs;
    std::vector<T> coeffs;

    int n = dof_to_ref[dof + 1] - dof_to_ref[dof];
    ref_dofs.resize(n);
    std::copy(std::next(ref_dofs_flat.begin(), dof_to_ref[dof]),
              std::next(ref_dofs_flat.begin(), dof_to_ref[dof + 1]),
              ref_dofs.begin());
    coeffs.resize(n);
    std::copy(std::next(ref_coeffs_flat.begin(), dof_to_ref[dof]),
              std::next(ref_coeffs_flat.begin(), dof_to_ref[dof + 1]),
              coeffs.begin());

    return {ref_dofs, coeffs};
  }

  /// @brief Replace constrained dofs with reference dofs in list
  /// @param dofs List of dofs, which may contain constrained dofs
  /// @return List of dofs, where constrained dofs are replaced with reference
  /// dofs
  std::vector<std::int32_t>
  modified_dofs(std::span<const std::int32_t> dofs) const
  {
    std::vector<std::int32_t> mdofs;
    mdofs.reserve(dofs.size());
    // Copy unconstrained dofs
    for (std::int32_t r : dofs)
    {
      if (dof_to_ref[r + 1] == dof_to_ref[r])
        mdofs.push_back(r);
    }
    // Add any new reference dofs
    for (std::int32_t r : dofs)
    {
      if (dof_to_ref[r + 1] != dof_to_ref[r])
      {
        for (int j = dof_to_ref[r]; j < dof_to_ref[r + 1]; ++j)
        {
          if (std::find(mdofs.begin(), mdofs.end(), ref_dofs_flat[j])
              == mdofs.end())
            mdofs.push_back(ref_dofs_flat[j]);
        }
      }
    }

    return mdofs;
  }

  /// @brief Compute K-matrix to convert between constrained and reference dofs
  /// @param dofs Input dofs
  /// @return Flattened K-matrix
  std::vector<T> Kmat(std::span<const std::int32_t> dofs) const
  {
    std::vector<std::int32_t> mdofs = modified_dofs(dofs);
    std::vector<T> mat(mdofs.size() * dofs.size(), 0.0);

    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      int r = dofs[i];
      if (dof_to_ref[r + 1] - dof_to_ref[r] == 0)
      {
        auto it = std::find(mdofs.begin(), mdofs.end(), r);
        assert(it != mdofs.end());
        int j = std::distance(mdofs.begin(), it);
        mat[mdofs.size() * i + j] = 1.0;
      }
      else
      {
        for (int k = dof_to_ref[r]; k < dof_to_ref[r + 1]; ++k)
        {
          auto it = std::find(mdofs.begin(), mdofs.end(), ref_dofs_flat[k]);
          assert(it != mdofs.end());
          int j = std::distance(mdofs.begin(), it);
          mat[mdofs.size() * i + j] = ref_coeffs_flat[k];
        }
      }
    }

    return mat;
  }

private:
  // Modified FunctionSpace with additional ghost dofs
  std::shared_ptr<const FunctionSpace<U>> _V;

  // Offset array to look up the reference data for constrained dofs
  // Contains pointers to ref_dofs_flat and ref_coeffs_flat, both of which have
  // the same layout. For unconstrained dofs, dof_to_ref[dof + 1] ==
  // dof_to_ref[dof]
  std::vector<std::int32_t> dof_to_ref;

  // Stored (local) indices of reference dofs, flattened
  std::vector<std::int32_t> ref_dofs_flat;
  // Stored coefficients to apply to reference dofs, flattened
  std::vector<T> ref_coeffs_flat;
};
} // namespace dolfinx::fem
