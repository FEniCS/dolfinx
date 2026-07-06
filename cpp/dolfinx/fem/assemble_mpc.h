
#pragma once

#include "Function.h"
#include "FunctionSpace.h"
#include "MPC.h"
#include "assembler.h"
#include "pack.h"
#include "traits.h"
#include "utils.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <cmath>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace dolfinx::fem
{

/// @brief Assemble bilinear form with a multipoint constraint into a matrix.
/// Matrix must already be initialised, with suitable sparsity.
/// Does not zero or finalise the matrix.
/// @param[in] mpc Multipoint constraint.
/// @param[in] mat_add The function for adding values into the matrix.
/// @param[in] a The bilinear form to assemble.
/// @param[in] bcs Dirichlet boundary conditions.
template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix_mpc(
    const fem::MPC<T, U>& mpc, auto mat_add, const fem::Form<T, U>& a,
    const std::vector<std::reference_wrapper<const DirichletBC<T, U>>>& bcs)
{
  // Check functionspace is the same for rows and cols
  if (a.function_spaces().size() != 2)
    throw std::runtime_error("Bilinear form required");

  // Ensure column and row space are the same
  if (a.function_spaces()[0].get() != a.function_spaces()[1].get())
    throw std::runtime_error("Different FunctionSpaces not supported.");

  // Check that DirichletBCs and MPC constraints do not conflict
  for (auto bc : bcs)
  {
    for (std::int32_t dof : bc.get().dof_indices().first)
    {
      spdlog::debug("BC dof {}", dof);
      if (mpc.constraints().num_links(dof) != 0)
        throw std::runtime_error("Clashing MPC constraint and DirichletBC");
    }
  }

  using cmdspan2_t = md::mdspan<const T, md::dextents<std::size_t, 2>>;
  using cmdspan2T_t
      = md::mdspan<const T, md::dextents<std::size_t, 2>, md::layout_left>;
  using mdspan2_t = md::mdspan<T, md::dextents<std::size_t, 2>>;

  // Debug helper: log a matrix with row/column dof labels, e.g. to compare
  // the unmodified element matrix against the MPC-modified A0/A1 stages.
  // Entries with |v| below a small tolerance are shown as 0, and all other
  // entries are rounded to 2 significant figures, to make the output easy
  // to scan by eye.
  auto debug_matrix
      = [](std::string_view name, std::span<const std::int32_t> row_dofs,
           std::span<const std::int32_t> col_dofs, std::span<const T> vals)
  {
    if constexpr (std::is_same_v<T, double> or std::is_same_v<T, float>)
    {
      if (!spdlog::default_logger()->should_log(spdlog::level::debug))
        return;

      constexpr T tol = static_cast<T>(1e-10);
      std::string out = fmt::format("{} ({} x {}):", name, row_dofs.size(),
                                    col_dofs.size());
      out += fmt::format("\n{:>10}", "");
      for (std::int32_t c : col_dofs)
        out += fmt::format(" {:>10}", c);
      for (std::size_t i = 0; i < row_dofs.size(); ++i)
      {
        out += fmt::format("\n{:>10}", row_dofs[i]);
        for (std::size_t j = 0; j < col_dofs.size(); ++j)
        {
          T v = vals[i * col_dofs.size() + j];
          out += std::abs(v) < tol ? fmt::format(" {:>10}", 0)
                                   : fmt::format(" {:>10.2g}", v);
        }
      }
      spdlog::debug("{}", out);
    }
  };

  auto mat_add_mpc =
      [mat_add, &mpc, &debug_matrix](std::span<const std::int32_t> rows,
                                      std::span<const std::int32_t> cols,
                                      std::span<const T> vals) mutable
  {
    int bs = mpc.V()->dofmap()->bs();

    // If no constraints, just add values to matrix
    int nc = 0;
    for (std::int32_t r : rows)
      for (int k = 0; k < bs; ++k)
        nc += mpc.constraints().num_links(r * bs + k);
    if (nc == 0)
    {
      mat_add(rows, cols, vals);
      return;
    }

    // Get list of block dofs required.
    std::vector<std::int32_t> dofs0;
    for (std::int32_t r : rows)
    {
      for (int k = 0; k < bs; ++k)
      {
        auto c = mpc.constraints().links(r * bs + k);
        if (c.empty())
          dofs0.push_back(r);
        else
        {
          for (auto [ref_dof, ref_coeff] : c)
            dofs0.push_back(ref_dof / bs);
        }
      }
    }
    // Remove duplicates and sort, expand to full dofs (including block size)
    std::sort(dofs0.begin(), dofs0.end());
    dofs0.erase(std::unique(dofs0.begin(), dofs0.end()), dofs0.end());
    std::vector<std::int32_t> dofs1;
    for (int d : dofs0)
    {
      for (int k = 0; k < bs; ++k)
        dofs1.push_back(d * bs + k);
    }

    if (spdlog::default_logger()->should_log(spdlog::level::debug))
    {
      // Unmodified element matrix, dof indices expanded by block size
      std::vector<std::int32_t> rows_bs(bs * rows.size());
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (int k = 0; k < bs; ++k)
          rows_bs[i * bs + k] = rows[i] * bs + k;
      std::vector<std::int32_t> cols_bs(bs * cols.size());
      for (std::size_t i = 0; i < cols.size(); ++i)
        for (int k = 0; k < bs; ++k)
          cols_bs[i * bs + k] = cols[i] * bs + k;
      debug_matrix("mat_add_mpc: unmodified", rows_bs, cols_bs, vals);
    }

    // Build a flattened map from each full (block-expanded) dof to its
    // position(s) in dofs1, with an associated coefficient: unconstrained
    // dofs map to themselves with coefficient 1, constrained dofs map to
    // their reference dofs. This lets both cases be handled uniformly
    // below, rather than branching on c.empty() in the hot loop.
    auto build_dof_map = [&](std::span<const std::int32_t> block_dofs)
    {
      std::vector<std::int32_t> offsets(1, 0);
      std::vector<std::int32_t> targets;
      std::vector<T> coeffs;
      for (std::int32_t d : block_dofs)
      {
        for (int k = 0; k < bs; ++k)
        {
          auto c = mpc.constraints().links(d * bs + k);
          if (c.empty())
          {
            auto it = std::lower_bound(dofs1.begin(), dofs1.end(), d * bs + k);
            if (it == dofs1.end() || *it != d * bs + k)
              throw std::runtime_error(
                  "Assembly: Unconstrained dof not found in dofs1");
            targets.push_back(std::distance(dofs1.begin(), it));
            coeffs.push_back(T(1));
          }
          else
          {
            for (auto [ref_dof, ref_coeff] : c)
            {
              auto it = std::lower_bound(dofs1.begin(), dofs1.end(), ref_dof);
              if (it == dofs1.end() || *it != ref_dof)
                throw std::runtime_error(
                    "Assembly: Reference dof not found in dofs1");
              targets.push_back(std::distance(dofs1.begin(), it));
              coeffs.push_back(ref_coeff);
            }
          }
          offsets.push_back(targets.size());
        }
      }
      return std::tuple(std::move(offsets), std::move(targets),
                         std::move(coeffs));
    };

    auto [row_off, row_tgt, row_coeff] = build_dof_map(rows);
    auto [col_off, col_tgt, col_coeff] = build_dof_map(cols);

    // Apply the row and column maps directly to the element matrix in a
    // single pass, i.e. A1 = Pr^T * vals * Pc, without materialising the
    // column-remapped intermediate (A0) matrix.
    std::vector<T> A1(dofs1.size() * dofs1.size(), T(0));
    std::size_t ncols_full = bs * cols.size();
    for (std::size_t r = 0; r < bs * rows.size(); ++r)
    {
      for (std::int32_t p = row_off[r]; p < row_off[r + 1]; ++p)
      {
        std::int32_t tr = row_tgt[p];
        T cr = row_coeff[p];
        for (std::size_t c = 0; c < ncols_full; ++c)
        {
          T val = cr * vals[r * ncols_full + c];
          for (std::int32_t q = col_off[c]; q < col_off[c + 1]; ++q)
            A1[tr * dofs1.size() + col_tgt[q]] += val * col_coeff[q];
        }
      }
    }

    // A1: both rows and columns remapped to dofs1
    debug_matrix("mat_add_mpc: A1", dofs1, dofs1, A1);

    // Revise rows, cols and vals for MPC
    mat_add(dofs0, dofs0, std::span<T>(A1.data(), A1.size()));
  };

  // Prepare constants and coefficients
  const std::vector<T> constants = pack_constants(a);
  auto coefficients = allocate_coefficient_storage(a);
  pack_coefficients(a, coefficients);

  // Main assembly
  spdlog::info("Assemble MPC");
  assemble_matrix(mat_add_mpc, a, bcs);

  int bs = mpc.V()->dofmap()->bs();
  spdlog::info("Apply MPC constraints, bs = {}", bs);
  // Insert constraint u_i = sum(a_j u_j)
  // N.B. assumes b_i = 0 (make sure this is done in RHS)

  // Check each dof individually, and add to matrix if it has constraints
  for (int dof = 0; dof < bs * mpc.V()->dofmap()->index_map->size_local();
       ++dof)
  {
    if (mpc.constraints().num_links(dof) == 0)
      continue;

    // Compile list of block dofs required.
    std::vector<std::int32_t> dofs0 = {dof / bs};
    for (auto [ref_dof, ref_coeff] : mpc.constraints().links(dof))
      dofs0.push_back(ref_dof / bs);
    std::sort(dofs0.begin(), dofs0.end());
    dofs0.erase(std::unique(dofs0.begin(), dofs0.end()), dofs0.end());

    // Expand dofs0 to include block size
    std::vector<std::int32_t> dofs1;
    for (int d : dofs0)
    {
      for (int k = 0; k < bs; ++k)
        dofs1.push_back(d * bs + k);
    }

    spdlog::debug("dof: {}", dof);
    if constexpr (std::is_same_v<T, double>)
    {
      for (auto [ref_dof, ref_coeff] : mpc.constraints().links(dof))
        spdlog::debug("  ref_dof: {}, ref_coeff: {}", ref_dof, ref_coeff);
    }
    spdlog::debug("dofs1: {}", fmt::join(dofs1, ", "));

    std::vector<T> v(dofs1.size() * dofs1.size(), T(0));

    // Find constrained dof in dofs1 and set diagonal to 1.0
    auto it = std::lower_bound(dofs1.begin(), dofs1.end(), dof);
    if (it == dofs1.end() || *it != dof)
      throw std::runtime_error("Constrained dof not found in dofs1");
    int jdof = std::distance(dofs1.begin(), it);
    v[jdof + dofs1.size() * jdof] = T(1.0);

    spdlog::debug("jdof: {}", jdof);

    // Find coefficients for each reference dof
    for (auto [ref_dof, ref_coeff] : mpc.constraints().links(dof))
    {
      auto it = std::lower_bound(dofs1.begin(), dofs1.end(), ref_dof);
      if (it == dofs1.end() || *it != ref_dof)
        throw std::runtime_error("Reference dof not found in dofs1");
      int m = std::distance(dofs1.begin(), it);
      if constexpr (std::is_same_v<T, double>)
        spdlog::debug("  m: {}, ref_coeff: {}", m, ref_coeff);
      v[m + dofs1.size() * jdof] = -ref_coeff;
    }

    // Transpose constraint row
    // for (std::size_t i = 0; i < dofs1.size(); ++i)
    //  v[i * dofs1.size() + jdof] = v[jdof * dofs1.size() + i];

    for (std::size_t i = 0; i < dofs1.size(); ++i)
    {
      if (i == jdof)
        continue;
      for (std::size_t j = 0; j < dofs1.size(); ++j)
        v[i * dofs1.size() + j]
            = v[jdof * dofs1.size() + i] * v[jdof * dofs1.size() + j];
    }

    mat_add(dofs0, dofs0, v);

    debug_matrix("assemble_matrix_mpc: constraint row", dofs1, dofs1, v);
  }
}

} // namespace dolfinx::fem
