
#pragma once

#include "Function.h"
#include "FunctionSpace.h"
#include "MPC.h"
#include "assembler.h"
#include "pack.h"
#include "traits.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <span>
#include <vector>

namespace dolfinx::fem
{

/// @brief Assemble bilinear form with a multipoint constraint into a matrix.
/// Matrix must already be initialised, with suitable sparsity.
/// Does not zero or finalise the matrix.
/// @param[in] mpcs Multipoint constraints for row and column spaces (row,
/// column).
/// @param[in] mat_add The function for adding values into the matrix.
/// @param[in] a The bilinear form to assemble.
/// @param[in] bcs Dirichlet boundary conditions.
template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix_mpc(
    std::array<std::reference_wrapper<const fem::MPC<T, U>>, 2> mpcs,
    auto mat_add, const fem::Form<T, U>& a,
    const std::vector<std::reference_wrapper<const DirichletBC<T, U>>>& bcs)
{
  if (a.function_spaces().size() != 2)
    throw std::runtime_error("Bilinear form required");

  const fem::MPC<T, U>& mpc_row = mpcs[0].get();
  const fem::MPC<T, U>& mpc_col = mpcs[1].get();

  if (a.function_spaces()[0].get() != mpc_row.V().get())
  {
    throw std::runtime_error(
        "Non-matching FunctionSpace on rows for Form and MPC");
  }
  if (a.function_spaces()[1].get() != mpc_col.V().get())
  {
    throw std::runtime_error(
        "Non-matching FunctionSpace on cols for Form and MPC");
  }

  // Check that DirichletBCs and MPC constraints do not conflict.
  // A dof that is both Dirichlet-constrained and MPC-constrained leads to
  // undefined behaviour: apply_lifting modifies the MPC-zeroed RHS row and
  // apply_mpc_solution overwrites the BC value post-solve.  Reject early
  // rather than silently producing wrong answers.
  for (auto bc : bcs)
  {
    if (bc.get().function_space().get() != mpc_col.V().get())
      throw std::runtime_error("BC not on column FunctionSpace.");
    for (std::int32_t dof : bc.get().dof_indices().first)
    {
      spdlog::debug("BC dof {}", dof);
      if (mpc_row.constraints().num_links(dof) != 0
          or mpc_col.constraints().num_links(dof) != 0)
      {
        throw std::runtime_error(
            "DirichletBC and MPC constraint on the same dof ("
            + std::to_string(dof)
            + ") — this combination is not supported. Apply the Dirichlet "
              "condition to a reference dof, or eliminate the constrained "
              "dof before constructing the MPC.");
      }
    }
  }

  const int bs_row = mpc_row.V()->dofmap()->bs();
  const int bs_col = mpc_col.V()->dofmap()->bs();

  // Debug helper: log a matrix with row/column dof labels, e.g. to
  // compare the unmodified element matrix against the MPC-modified A0/A1
  // stages. Entries with |v| below a small tolerance are shown as 0, and
  // all other entries are rounded to 2 significant figures, to make the
  // output easy to scan by eye.
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

  auto mat_add_mpc
      = [mat_add, &mpc_row, &mpc_col, &debug_matrix, bs_row, bs_col](
            std::span<const std::int32_t> rows,
            std::span<const std::int32_t> cols, std::span<const T> vals) mutable
  {
    // If no constraints, just add values to matrix
    int nc = 0;
    for (std::int32_t r : rows)
      for (int k = 0; k < bs_row; ++k)
        nc += mpc_row.constraints().num_links(r * bs_row + k);
    for (std::int32_t c : cols)
      for (int k = 0; k < bs_col; ++k)
        nc += mpc_col.constraints().num_links(c * bs_col + k);
    if (nc == 0)
    {
      mat_add(rows, cols, vals);
      return;
    }

    if (spdlog::default_logger()->should_log(spdlog::level::debug))
    {
      // Unmodified element matrix, dof indices expanded by block size
      std::vector<std::int32_t> rows_bs(bs_row * rows.size());
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (int k = 0; k < bs_row; ++k)
          rows_bs[i * bs_row + k] = rows[i] * bs_row + k;
      std::vector<std::int32_t> cols_bs(bs_col * cols.size());
      for (std::size_t i = 0; i < cols.size(); ++i)
        for (int k = 0; k < bs_col; ++k)
          cols_bs[i * bs_col + k] = cols[i] * bs_col + k;
      debug_matrix("mat_add_mpc: unmodified", rows_bs, cols_bs, vals);
    }

    // Build a flattened map from each full (block-expanded) dof to its
    // expanded block position(s), with an associated coefficient: unconstrained
    // dofs map to themselves with coefficient 1, constrained dofs map to
    // their reference dofs. This lets both cases be handled uniformly
    // below, rather than branching on c.empty() in the hot loop.
    auto build_dof_map
        = [](std::span<const std::int32_t> indices, const MPC<T, U>& mpc)
    {
      int bs = mpc.V()->dofmap()->bs();

      // Resolve each block-expanded dof to its reference dof(s): unconstrained
      // maps to itself (coeff 1), constrained maps to its reference dofs.
      std::vector<std::int32_t> offsets = {0};
      std::vector<std::int32_t> refs;
      std::vector<T> coeffs;
      for (std::int32_t d : indices)
      {
        for (int k = 0; k < bs; ++k)
        {
          std::int32_t dof = d * bs + k;
          auto c = mpc.constraints().links(dof);
          if (c.empty())
          {
            refs.push_back(dof);
            coeffs.push_back(T(1));
          }
          else
          {
            for (auto [ref_dof, ref_coeff] : c)
            {
              refs.push_back(ref_dof);
              coeffs.push_back(ref_coeff);
            }
          }
          offsets.push_back(refs.size());
        }
      }

      // Unique, sorted block dofs referenced.
      std::vector<std::int32_t> dofs0(refs.size());
      std::transform(refs.begin(), refs.end(), dofs0.begin(),
                     [bs](std::int32_t r) { return r / bs; });
      std::sort(dofs0.begin(), dofs0.end());
      dofs0.erase(std::unique(dofs0.begin(), dofs0.end()), dofs0.end());

      // Map each reference dof to its expanded block position.
      std::vector<std::int32_t> targets(refs.size());
      for (std::size_t i = 0; i < refs.size(); ++i)
      {
        int component = refs[i] % bs;
        auto it = std::lower_bound(dofs0.begin(), dofs0.end(), refs[i] / bs);
        if (it == dofs0.end() || *it != refs[i] / bs)
          throw std::runtime_error(
              "Assembly: Reference dof not found in dofs0");
        targets[i] = std::distance(dofs0.begin(), it) * bs + component;
      }

      return std::tuple(std::move(offsets), std::move(targets),
                        std::move(coeffs), std::move(dofs0));
    };

    auto [row_off, row_tgt, row_coeff, dofs0_row]
        = build_dof_map(rows, mpc_row);
    auto [col_off, col_tgt, col_coeff, dofs0_col]
        = build_dof_map(cols, mpc_col);

    // Apply the row and column maps directly to the element matrix in a
    // single pass, i.e. Ae = Pr^T * vals * Pc, without materialising the
    // column-remapped intermediate (A0) matrix.
    std::vector<T> Ae(dofs0_row.size() * bs_row * dofs0_col.size() * bs_col,
                      T(0));
    std::size_t ncols_full = bs_col * cols.size();
    for (std::size_t r = 0; r < bs_row * rows.size(); ++r)
    {
      for (std::int32_t p = row_off[r]; p < row_off[r + 1]; ++p)
      {
        std::int32_t tr = row_tgt[p];
        T cr = row_coeff[p];
        for (std::size_t c = 0; c < ncols_full; ++c)
        {
          T val = cr * vals[r * ncols_full + c];
          for (std::int32_t q = col_off[c]; q < col_off[c + 1]; ++q)
            Ae[tr * dofs0_col.size() * bs_col + col_tgt[q]]
                += val * col_coeff[q];
        }
      }
    }

    if (spdlog::default_logger()->should_log(spdlog::level::debug))
    {
      // A1: both rows and columns remapped to their reference dofs.
      // Expand dofs0_row/dofs0_col by their respective block sizes purely
      // for labelling the debug output.
      std::vector<std::int32_t> dofs1_row;
      dofs1_row.reserve(dofs0_row.size() * bs_row);
      for (std::int32_t d : dofs0_row)
        for (int k = 0; k < bs_row; ++k)
          dofs1_row.push_back(d * bs_row + k);

      std::vector<std::int32_t> dofs1_col;
      dofs1_col.reserve(dofs0_col.size() * bs_col);
      for (std::int32_t d : dofs0_col)
        for (int k = 0; k < bs_col; ++k)
          dofs1_col.push_back(d * bs_col + k);

      debug_matrix("mat_add_mpc: Ae", dofs1_row, dofs1_col, Ae);
    }

    // Revise rows, cols and vals for MPC
    mat_add(dofs0_row, dofs0_col, std::span<T>(Ae.data(), Ae.size()));
  };

  // Prepare constants and coefficients
  const std::vector<T> constants = pack_constants(a);
  auto coefficients = allocate_coefficient_storage(a);
  pack_coefficients(a, coefficients);

  // Main assembly
  spdlog::debug("Assemble MPC");
  assemble_matrix(mat_add_mpc, a, bcs);

  // Set diagonal = 1 for each locally-owned constrained dof so the row is
  // non-singular.  The P^T A P step has already zeroed these rows/columns,
  // so no other entry needs touching here.  The correct solution value
  // u[i] = sum(c_k * u[ref_k]) is recovered by apply_mpc_solution after
  // the linear solve.
  spdlog::debug("Apply MPC diagonal, bs = {}", bs_row);
  for (int dof = 0;
       dof < bs_row * mpc_row.V()->dofmap()->index_map->size_local(); ++dof)
  {
    if (mpc_row.constraints().num_links(dof) == 0)
      continue;

    std::int32_t block_dof = dof / bs_row;
    int component = dof % bs_row;
    std::vector<T> v(bs_row * bs_row, T(0));
    v[component * bs_row + component] = T(1);
    mat_add(std::span<const std::int32_t>({&block_dof, 1}),
            std::span<const std::int32_t>({&block_dof, 1}), v);
  }
}

// ============================================================
// RHS / vector assembly
// ============================================================

/// @brief Apply MPC constraints to an assembled vector.
///
/// For each locally-owned constrained dof @p i this function:
///   1. Distributes @p b[i] to every reference dof via
///      `b[ref_k] += c_k * b[i]`  (the Pᵀ b step).
///   2. Sets `b[i] = 0`  (homogeneous constraint row RHS).
///
/// Reference dof indices stored in `mpc.constraints()` are already local
/// indices in the extended IndexMap of `mpc.V()`, so the write to
/// `b[ref_k]` is correct whether `ref_k` is an owned dof or an extra ghost
/// added by the MPC.
///
/// @pre  scatter_rev (ghost → owner accumulation) must already have been
///       applied to @p b so that each constrained dof slot holds the full
///       assembled value.  Constrained dofs commonly sit on process
///       boundaries (e.g. periodic BCs, tied interfaces), so this scatter
///       should be treated as always required.
///
/// @post A further scatter_rev is required after this call to push the
///       Pᵀ contributions that landed in ghost slots back to their owning
///       processes.
///
/// @note For inhomogeneous constraints (u_i = Σ c_k u_j_k + g_i), the
///       constant g_i is read from `mpc.constants()` and placed in
///       `b[dof]` so the linear system RHS enforces the inhomogeneous value.
///
/// @param[in,out] b  Assembled vector using the extended IndexMap of
///                   `mpc.V()`.  Size must be at least
///                   `index_map_bs * (size_local + num_ghosts)`.
/// @param[in]    mpc Multipoint constraint.
template <dolfinx::scalar T, std::floating_point U>
void apply_mpc_vector(std::span<T> b, const MPC<T, U>& mpc)
{
  const auto& C = mpc.constraints();
  const auto& K = mpc.constants();
  const std::int32_t index_map_bs = mpc.V()->dofmap()->index_map_bs();
  const std::int32_t num_owned
      = mpc.V()->dofmap()->index_map->size_local() * index_map_bs;

  for (std::int32_t dof = 0; dof < num_owned; ++dof)
  {
    auto links = C.links(dof);
    auto clinks = K.links(dof);
    if (links.empty() and clinks.empty())
      continue;

    // P^T step: distribute b[constrained] to each reference dof.
    const T b_constrained = b[dof];
    for (auto [ref_dof, coeff] : links)
      b[ref_dof] += coeff * b_constrained;

    // Constrained row: for a homogeneous constraint g_i = 0;
    // for inhomogeneous constraints sum the constant contributions.
    b[dof] = T(0);
    for (auto c : clinks)
      b[dof] += c;
  }
}

/// @brief Recover constrained dof values from the solution after a linear
/// solve.
///
/// After solving the linear system assembled with assemble_matrix_mpc, the
/// constrained dof slots hold whatever the solver placed there (typically 0,
/// since those rows were set to a unit diagonal with zero RHS).  This
/// function overwrites each such slot with the correct constraint value:
///
///   u[i] ← Σ_k c_k u[ref_k]
///
/// @note In parallel, scatter_fwd must be called on @p u before this
///       function so that ghost values of reference dofs are current.
///
/// @param[in,out] u  Solution vector (extended IndexMap of mpc.V()).
/// @param[in]    mpc Multipoint constraint.
template <dolfinx::scalar T, std::floating_point U>
void apply_mpc_solution(std::span<T> u, const MPC<T, U>& mpc)
{
  const auto& C = mpc.constraints();
  const auto& K = mpc.constants();
  const std::int32_t index_map_bs = mpc.V()->dofmap()->index_map_bs();
  const std::int32_t num_owned
      = mpc.V()->dofmap()->index_map->size_local() * index_map_bs;

  for (std::int32_t dof = 0; dof < num_owned; ++dof)
  {
    auto links = C.links(dof);
    auto clinks = K.links(dof);
    if (links.empty() and clinks.empty())
      continue;
    u[dof] = T(0);
    for (auto [ref_dof, coeff] : links)
      u[dof] += coeff * u[ref_dof];
    for (auto c : clinks)
      u[dof] += c;
  }
}

/// @brief Apply MPC constraints to a nonlinear residual vector.
///
/// Variant of apply_mpc_vector for Newton iteration, where the constrained
/// row must hold the constraint *residual* rather than zero:
///
///   F[i] ← u[i] − Σ_k c_k u[ref_k]
///
/// and the reference dof rows still receive the Pᵀ distribution of F[i]
/// before it is overwritten.  The rest of the residual (non-constrained
/// dofs) is unchanged.
///
/// @note The standard sequence for a nonlinear residual step is:
/// @code
///   assemble_vector(F, L);
///   index_map.scatter_rev(F, std::plus<T>());  // constrained dofs complete
///   apply_mpc_residual(F, u, mpc);
///   index_map.scatter_rev(F, std::plus<T>());  // accumulate P^T ghosts
/// @endcode
///
/// @param[in,out] F  Residual vector (extended IndexMap of mpc.V()).
/// @param[in]    u   Current solution vector (extended IndexMap, same size).
/// @param[in]    mpc Multipoint constraint.
template <dolfinx::scalar T, std::floating_point U>
void apply_mpc_residual(std::span<T> F, std::span<const T> u,
                        const MPC<T, U>& mpc)
{
  const auto& C = mpc.constraints();
  const std::int32_t index_map_bs = mpc.V()->dofmap()->index_map_bs();
  const std::int32_t num_owned
      = mpc.V()->dofmap()->index_map->size_local() * index_map_bs;

  for (std::int32_t dof = 0; dof < num_owned; ++dof)
  {
    auto links = C.links(dof);
    if (links.empty())
      continue;

    // P^T step: distribute the assembled residual F[constrained] to
    // reference dofs before overwriting it.
    const T F_constrained = F[dof];
    for (auto [ref_dof, coeff] : links)
      F[ref_dof] += coeff * F_constrained;

    // Constrained row: constraint residual u_i − Σ c_k u_{ref_k}.
    T constraint_res = u[dof];
    for (auto [ref_dof, coeff] : links)
      constraint_res -= coeff * u[ref_dof];
    F[dof] = constraint_res;
  }
}

} // namespace dolfinx::fem
