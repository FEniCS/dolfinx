
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
/// @param[in] mpc Multipoint constraints for row and column spaces.
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

  // Check that DirichletBCs and MPC constraints do not conflict
  for (auto bc : bcs)
  {
    if (bc.get().function_space().get() != mpc_col.V().get())
      throw std::runtime_error("BC not on column FunctionSpace.");
    for (std::int32_t dof : bc.get().dof_indices().first)
    {
      spdlog::debug("BC dof {}", dof);
      if (mpc_col.constraints().num_links(dof) != 0)
        throw std::runtime_error("Clashing MPC constraint and DirichletBC");
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
      // maps to itself (coeff 1), constrained maps to its masters.
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
  spdlog::info("Assemble MPC");
  assemble_matrix(mat_add_mpc, a, bcs);

  // If different spaces, skip this step
  if (a.function_spaces()[0].get() != a.function_spaces()[1].get())
    return;

  spdlog::info("Apply MPC constraints, bs = {}", bs_row);
  // Insert constraint u_i = sum(a_j u_j)
  // N.B. assumes b_i = 0 (make sure this is done in RHS)

  // Check each dof individually, and add to matrix if it has constraints
  for (int dof = 0;
       dof < bs_row * mpc_row.V()->dofmap()->index_map->size_local(); ++dof)
  {
    if (mpc_row.constraints().num_links(dof) == 0)
      continue;

    // Compile list of block dofs required.
    std::vector<std::int32_t> dofs0 = {dof / bs_row};
    for (auto [ref_dof, ref_coeff] : mpcs[0].get().constraints().links(dof))
      dofs0.push_back(ref_dof / bs_row);
    std::sort(dofs0.begin(), dofs0.end());
    dofs0.erase(std::unique(dofs0.begin(), dofs0.end()), dofs0.end());

    // Expand dofs0 to include block size
    std::vector<std::int32_t> dofs1;
    for (int d : dofs0)
    {
      for (int k = 0; k < bs_row; ++k)
        dofs1.push_back(d * bs_row + k);
    }

    spdlog::debug("dof: {}", dof);

    std::vector<T> v(dofs1.size() * dofs1.size(), T(0));
    // Find constrained dof in dofs1 and set diagonal to 1.0
    auto it = std::lower_bound(dofs1.begin(), dofs1.end(), dof);
    if (it == dofs1.end() || *it != dof)
      throw std::runtime_error("Constrained dof not found in dofs1");
    std::size_t jdof = std::distance(dofs1.begin(), it);
    v[jdof + dofs1.size() * jdof] = T(1.0);

    spdlog::debug("jdof: {}", jdof);

    // Find coefficients for each reference dof
    for (auto [ref_dof, ref_coeff] : mpc_row.constraints().links(dof))
    {
      auto it = std::lower_bound(dofs1.begin(), dofs1.end(), ref_dof);
      if (it == dofs1.end() || *it != ref_dof)
        throw std::runtime_error("Reference dof not found in dofs1");
      std::size_t m = std::distance(dofs1.begin(), it);
      if constexpr (std::is_same_v<T, double>)
        spdlog::debug("  m: {}, ref_coeff: {}", m, ref_coeff);
      v[m + dofs1.size() * jdof] = -ref_coeff;
    }

    for (std::size_t i = 0; i < dofs1.size(); ++i)
    {
      if (i == jdof)
        continue;
      for (std::size_t j = 0; j < dofs1.size(); ++j)
      {
        v[i * dofs1.size() + j]
            = v[jdof * dofs1.size() + i] * v[jdof * dofs1.size() + j];
      }
    }

    mat_add(dofs0, dofs0, v);

    debug_matrix("assemble_matrix_mpc: constraint row", dofs1, dofs1, v);
  }
}

// ============================================================
// RHS / vector assembly
// ============================================================

/// @brief Apply MPC constraints to an assembled vector.
///
/// For each locally-owned constrained (slave) dof @p i this function:
///   1. Distributes @p b[i] to every reference dof via
///      `b[ref_k] += c_k * b[i]`  (the Pᵀ b step).
///   2. Sets `b[i] = 0`  (homogeneous constraint row RHS).
///
/// Reference dof indices stored in `mpc.constraints()` are already local
/// indices in the extended IndexMap of `mpc.V()`, so the write to
/// `b[ref_k]` is correct whether `ref_k` is an owned dof or an extra ghost
/// added by the MPC.
///
/// @pre  A scatter_rev (ghost → owner accumulation) must already have been
///       applied to @p b so that each slave dof slot holds the full assembled
///       value.  (In the common case where slave dofs are interior to a single
///       process this scatter is not strictly necessary, but it is needed for
///       correctness when a slave dof lies on a process boundary.)
///
/// @post A further scatter_rev is required after this call to push the
///       Pᵀ contributions that landed in ghost slots back to their owning
///       processes.
///
/// @note For inhomogeneous constraints (u_i = Σ c_k u_j_k + g_i), replace
///       `b[dof] = T(0)` with `b[dof] = g_i` once the constant term is
///       stored in MPC (currently a @todo in MPC.h).
///
/// @param[in,out] b  Assembled vector using the extended IndexMap of
///                   `mpc.V()`.  Size must be at least
///                   `index_map_bs * (size_local + num_ghosts)`.
/// @param[in]    mpc Multipoint constraint.
template <dolfinx::scalar T, std::floating_point U>
void apply_mpc_vector(std::span<T> b, const MPC<T, U>& mpc)
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

    // P^T step: distribute b[slave] to each reference dof.
    const T b_slave = b[dof];
    for (auto [ref_dof, coeff] : links)
      b[ref_dof] += coeff * b_slave;

    // Slave row: set to 0 for a homogeneous constraint.
    // TODO: for inhomogeneous constraints set b[dof] = g_i (the constant
    // stored per slave dof in MPC::_constants, not yet implemented).
    b[dof] = T(0);
  }
}

/// @brief Apply MPC constraints to a nonlinear residual vector.
///
/// Variant of apply_mpc_vector for Newton iteration, where the slave row
/// must hold the constraint *residual* rather than zero:
///
///   F[i] ← u[i] − Σ_k c_k u[ref_k]
///
/// and the reference dof rows still receive the Pᵀ distribution of F[i]
/// before it is overwritten.  The rest of the residual (non-slave dofs) is
/// unchanged.
///
/// @note The standard sequence for a nonlinear residual step is:
/// @code
///   assemble_vector(F.span(), L);          // standard residual assembly
///   F.scatter_rev(std::plus<T>());         // accumulate ghosts
///   apply_mpc_residual(F.span(), u.span(), mpc);
///   F.scatter_rev(std::plus<T>());         // accumulate P^T ghost contributions
/// @endcode
///
/// @param[in,out] F  Residual vector (extended IndexMap).
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

    // P^T step: distribute the assembled residual F[slave] to reference dofs
    // before overwriting it.
    const T F_slave = F[dof];
    for (auto [ref_dof, coeff] : links)
      F[ref_dof] += coeff * F_slave;

    // Slave row: constraint residual u_i − Σ c_k u_{ref_k}.
    T constraint_res = u[dof];
    for (auto [ref_dof, coeff] : links)
      constraint_res -= coeff * u[ref_dof];
    F[dof] = constraint_res;
  }
}

/// @brief Assemble a linear form with MPC constraints into a vector.
///
/// Calls the standard assemble_vector followed by apply_mpc_vector.  Because
/// there is no insertion lambda for vectors (unlike matrices), the MPC
/// transformation is applied as a post-processing step.
///
/// @note The recommended parallel usage is:
/// @code
///   // 1. Assemble (fills owned and ghost slots).
///   std::fill(b.begin(), b.end(), T(0));
///   assemble_vector_mpc(b, L, mpc);
///
///   // 2. Single scatter_rev accumulates both the standard ghost
///   //    contributions and the P^T ghost contributions from apply_mpc_vector.
///   mpc.V()->dofmap()->index_map->scatter_rev(b_span, std::plus<T>());
///
///   // 3. Apply Dirichlet BCs if any.
///   fem::set_bc(b, bcs);
/// @endcode
///
/// @note In the (unusual) case where a slave dof lies on a process boundary
///       (i.e. it is a ghost on a neighbouring process that contributes to
///       its assembled value), a scatter_rev should be called between
///       assemble_vector and apply_mpc_vector.  For interior slave dofs —
///       the common case — the single post-assembly scatter_rev is sufficient.
///
/// @param[in,out] b  Pre-zeroed vector (extended IndexMap of mpc.V()).
/// @param[in]    L   Linear form.
/// @param[in]    mpc Multipoint constraint.
template <dolfinx::scalar T, std::floating_point U>
void assemble_vector_mpc(std::span<T> b, const Form<T, U>& L,
                         const MPC<T, U>& mpc)
{
  if (L.function_spaces().size() != 1)
    throw std::runtime_error("assemble_vector_mpc: linear form required");

  if (L.function_spaces()[0].get() != mpc.V().get())
    throw std::runtime_error(
        "assemble_vector_mpc: form function space does not match mpc.V()");

  // Standard vector assembly using the extended function space.
  assemble_vector(b, L);

  // P^T step: distribute slave contributions to reference dofs and
  // zero slave rows.
  apply_mpc_vector(b, mpc);
}


} // namespace dolfinx::fem
