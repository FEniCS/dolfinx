// Copyright (C) 2007-2021 Michal Habera, Anders Logg, Garth N. Wells
// and Jørgen S.Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "DofMap.h"
#include "Function.h"
#include "FunctionSpace.h"
#include <array>
#include <concepts>
#include <dolfinx/common/types.h>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace dolfinx::fem
{

/// Find degrees-of-freedom which belong to the provided mesh entities
/// (topological). Note that degrees-of-freedom for discontinuous
/// elements are associated with the cell even if they may appear to be
/// associated with a facet/edge/vertex.
///
/// @param[in] topology Mesh topology.
/// @param[in] dofmap Dofmap that associated DOFs with cells.
/// @param[in] dim Topological dimension of mesh entities on which
/// degrees-of-freedom will be located
/// @param[in] entities Indices of mesh entities. All DOFs associated
/// with the closure of these indices will be returned
/// @param[in] remote True to return also "remotely located"
/// degree-of-freedom indices. Remotely located degree-of-freedom
/// indices are local/owned by the current process, but which the
/// current process cannot identify because it does not recognize mesh
/// entity as a marked. For example, a boundary condition dof at a
/// vertex where this process does not have the associated boundary
/// facet. This commonly occurs with partitioned meshes.
/// @return Array of DOF index blocks (local to the MPI rank) in the
/// space V. The array uses the block size of the dofmap associated
/// with V.
std::vector<std::int32_t>
locate_dofs_topological(mesh::Topology& topology, const DofMap& dofmap, int dim,
                        std::span<const std::int32_t> entities,
                        bool remote = true);

/// Find degrees-of-freedom which belong to the provided mesh entities
/// (topological). Note that degrees-of-freedom for discontinuous
/// elements are associated with the cell even if they may appear to be
/// associated with a facet/edge/vertex.
///
/// @param[in] topology Mesh topology.
/// @param[in] dofmaps The dofmaps.
/// @param[in] dim Topological dimension of mesh entities on which
/// degrees-of-freedom will be located
/// @param[in] entities Indices of mesh entities. All DOFs associated
/// with the closure of these indices will be returned
/// @param[in] remote True to return also "remotely located"
/// degree-of-freedom indices. Remotely located degree-of-freedom
/// indices are local/owned by the current process, but which the
/// current process cannot identify because it does not recognize mesh
/// entity as a marked. For example, a boundary condition dof at a
/// vertex where this process does not have the associated boundary
/// facet. This commonly occurs with partitioned meshes.
/// @return Array of DOF indices (local to the MPI rank) in the spaces
/// V[0] and V[1]. The array[0](i) entry is the DOF index in the space
/// V[0] and array[1](i) is the corresponding DOF entry in the space
/// V[1]. The returned dofs are 'unrolled', i.e. block size = 1.
std::array<std::vector<std::int32_t>, 2> locate_dofs_topological(
    mesh::Topology& topology,
    std::array<std::reference_wrapper<const DofMap>, 2> dofmaps, int dim,
    std::span<const std::int32_t> entities, bool remote = true);

/// @brief Find degrees of freedom whose geometric coordinate is true
/// for the provided marking function.
///
/// @attention This function is slower than the topological version.
///
/// @param[in] V The function (sub)space on which degrees of freedom
/// will be located.
/// @param[in] marker_fn Function marking tabulated degrees of freedom
/// @return Array of DOF index blocks (local to the MPI rank) in the
/// space V. The array uses the block size of the dofmap associated
/// with V.
template <std::floating_point T, typename U>
std::vector<std::int32_t> locate_dofs_geometrical(const FunctionSpace<T>& V,
                                                  U marker_fn)
{
  // FIXME: Calling V.tabulate_dof_coordinates() is very expensive,
  // especially when we usually want the boundary dofs only. Add
  // interface that computes dofs coordinates only for specified cell.

  assert(V.element());
  if (V.element()->is_mixed())
  {
    throw std::runtime_error(
        "Cannot locate dofs geometrically for mixed space. Use subspaces.");
  }

  // Compute dof coordinates
  const std::vector<T> dof_coordinates = V.tabulate_dof_coordinates(true);

  namespace stdex = std::experimental;
  using cmdspan3x_t
      = stdex::mdspan<const T,
                      stdex::extents<std::size_t, 3, stdex::dynamic_extent>>;

  // Compute marker for each dof coordinate
  cmdspan3x_t x(dof_coordinates.data(), 3, dof_coordinates.size() / 3);
  const std::vector<std::int8_t> marked_dofs = marker_fn(x);

  std::vector<std::int32_t> dofs;
  dofs.reserve(std::count(marked_dofs.begin(), marked_dofs.end(), true));
  for (std::size_t i = 0; i < marked_dofs.size(); ++i)
  {
    if (marked_dofs[i])
      dofs.push_back(i);
  }

  return dofs;
}

/// Finds degrees of freedom whose geometric coordinate is true for the
/// provided marking function.
///
/// @attention This function is slower than the topological version
///
/// @param[in] V The function (sub)space(s) on which degrees of freedom
/// will be located. The spaces must share the same mesh and element
/// type.
/// @param[in] marker_fn Function marking tabulated degrees of freedom
/// @return Array of DOF indices (local to the MPI rank) in the spaces
/// V[0] and V[1]. The array[0](i) entry is the DOF index in the space
/// V[0] and array[1](i) is the corresponding DOF entry in the space
/// V[1]. The returned dofs are 'unrolled', i.e. block size = 1.
template <std::floating_point T, typename U>
std::array<std::vector<std::int32_t>, 2> locate_dofs_geometrical(
    const std::array<std::reference_wrapper<const FunctionSpace<T>>, 2>& V,
    U marker_fn)
{
  // FIXME: Calling V.tabulate_dof_coordinates() is very expensive,
  // especially when we usually want the boundary dofs only. Add
  // interface that computes dofs coordinates only for specified cell.

  // Get function spaces
  const FunctionSpace<T>& V0 = V.at(0).get();
  const FunctionSpace<T>& V1 = V.at(1).get();

  // Get mesh
  auto mesh = V0.mesh();
  assert(mesh);
  assert(V1.mesh());
  if (mesh != V1.mesh())
    throw std::runtime_error("Meshes are not the same.");
  const int tdim = mesh->topology()->dim();

  assert(V0.element());
  assert(V1.element());
  if (*V0.element() != *V1.element())
    throw std::runtime_error("Function spaces must have the same element.");

  // Compute dof coordinates
  const std::vector<T> dof_coordinates = V1.tabulate_dof_coordinates(true);

  namespace stdex = std::experimental;
  using cmdspan3x_t
      = stdex::mdspan<const T,
                      stdex::extents<std::size_t, 3, stdex::dynamic_extent>>;

  // Evaluate marker for each dof coordinate
  cmdspan3x_t x(dof_coordinates.data(), 3, dof_coordinates.size() / 3);
  const std::vector<std::int8_t> marked_dofs = marker_fn(x);

  // Get dofmaps
  std::shared_ptr<const DofMap> dofmap0 = V0.dofmap();
  assert(dofmap0);
  const int bs0 = dofmap0->bs();
  std::shared_ptr<const DofMap> dofmap1 = V1.dofmap();
  assert(dofmap1);
  const int bs1 = dofmap1->bs();

  const int element_bs = dofmap0->element_dof_layout().block_size();
  assert(element_bs == dofmap1->element_dof_layout().block_size());

  // Iterate over cells
  auto topology = mesh->topology();
  assert(topology);
  std::vector<std::array<std::int32_t, 2>> bc_dofs;
  for (int c = 0; c < topology->connectivity(tdim, 0)->num_nodes(); ++c)
  {
    // Get cell dofmaps
    auto cell_dofs0 = dofmap0->cell_dofs(c);
    auto cell_dofs1 = dofmap1->cell_dofs(c);

    // Loop over cell dofs and add to bc_dofs if marked.
    for (std::size_t i = 0; i < cell_dofs1.size(); ++i)
    {
      if (marked_dofs[cell_dofs1[i]])
      {
        // Unroll over blocks
        for (int k = 0; k < element_bs; ++k)
        {
          const int local_pos = element_bs * i + k;
          const std::div_t pos0 = std::div(local_pos, bs0);
          const std::div_t pos1 = std::div(local_pos, bs1);
          const std::int32_t dof_index0
              = bs0 * cell_dofs0[pos0.quot] + pos0.rem;
          const std::int32_t dof_index1
              = bs1 * cell_dofs1[pos1.quot] + pos1.rem;
          bc_dofs.push_back({dof_index0, dof_index1});
        }
      }
    }
  }

  // Remove duplicates
  std::sort(bc_dofs.begin(), bc_dofs.end());
  bc_dofs.erase(std::unique(bc_dofs.begin(), bc_dofs.end()), bc_dofs.end());

  // Copy to separate array
  std::array dofs = {std::vector<std::int32_t>(bc_dofs.size()),
                     std::vector<std::int32_t>(bc_dofs.size())};
  std::transform(bc_dofs.cbegin(), bc_dofs.cend(), dofs[0].begin(),
                 [](auto dof) { return dof[0]; });
  std::transform(bc_dofs.cbegin(), bc_dofs.cend(), dofs[1].begin(),
                 [](auto dof) { return dof[1]; });

  return dofs;
}

/// Object for setting (strong) Dirichlet boundary conditions
///
///     \f$u = g \ \text{on} \ G\f$,
///
/// where \f$u\f$ is the solution to be computed, \f$g\f$ is a function
/// and \f$G\f$ is a sub domain of the mesh.
///
/// A DirichletBC is specified by the function \f$g\f$, the function
/// space (trial space) and degrees of freedom to which the boundary
/// condition applies.
template <typename T, std::floating_point U = dolfinx::scalar_value_type_t<T>>
class DirichletBC
{
private:
  /// Compute number of owned dofs indices. Will contain 'gaps' for
  /// sub-spaces.
  std::size_t num_owned(const DofMap& dofmap,
                        std::span<const std::int32_t> dofs)
  {
    int bs = dofmap.index_map_bs();
    std::int32_t map_size = dofmap.index_map->size_local();
    std::int32_t owned_size = bs * map_size;
    auto it = std::lower_bound(dofs.begin(), dofs.end(), owned_size);
    return std::distance(dofs.begin(), it);
  }

  /// Unroll dofs for block size.
  static std::vector<std::int32_t>
  unroll_dofs(std::span<const std::int32_t> dofs, int bs)
  {
    std::vector<std::int32_t> dofs_unrolled(bs * dofs.size());
    for (std::size_t i = 0; i < dofs.size(); ++i)
      for (int k = 0; k < bs; ++k)
        dofs_unrolled[bs * i + k] = bs * dofs[i] + k;
    return dofs_unrolled;
  }

public:
  /// @brief Create a representation of a Dirichlet boundary condition
  /// constrained by a scalar- or vector-valued constant.
  ///
  /// @pre `dofs` must be sorted.
  ///
  /// @param[in] g The boundary condition value (`T` or convertible to
  /// `std::span<const T>`)
  /// @param[in] dofs Degree-of-freedom block indices to be constrained.
  /// The indices must be sorted.
  /// @param[in] V The function space to be constrained
  /// @note Can be used only with point-evaluation elements.
  /// @note The indices in `dofs` are for *blocks*, e.g. a block index
  /// corresponds to 3 degrees-of-freedom if the dofmap associated with
  /// `g` has block size 3.
  /// @note The size of of `g` must be equal to the block size if `V`.
  /// Use the Function version if this is not the case, e.g. for some
  /// mixed spaces.
  template <typename S, std::convertible_to<std::vector<std::int32_t>> X,
            typename
            = std::enable_if_t<std::is_convertible_v<S, T>
                               or std::is_convertible_v<S, std::span<const T>>>>
  DirichletBC(const S& g, X&& dofs, std::shared_ptr<const FunctionSpace<U>> V)
      : DirichletBC(std::make_shared<Constant<T>>(g), dofs, V)
  {
  }

  /// @brief Create a representation of a Dirichlet boundary condition
  /// constrained by a fem::Constant.
  ///
  ///@pre `dofs` must be sorted.
  ///
  /// @param[in] g The boundary condition value.
  /// @param[in] dofs Degree-of-freedom block indices to be constrained.
  /// @param[in] V The function space to be constrained
  /// @note Can be used only with point-evaluation elements.
  /// @note The indices in `dofs` are for *blocks*, e.g. a block index
  /// corresponds to 3 degrees-of-freedom if the dofmap associated with
  /// `g` has block size 3.
  /// @note The size of of `g` must be equal to the block size if `V`.
  /// Use the Function version if this is not the case, e.g. for some
  /// mixed spaces.
  template <std::convertible_to<std::vector<std::int32_t>> X>
  DirichletBC(std::shared_ptr<const Constant<T>> g, X&& dofs,
              std::shared_ptr<const FunctionSpace<U>> V)
      : _function_space(V), _g(g), _dofs0(std::forward<X>(dofs)),
        _owned_indices0(num_owned(*V->dofmap(), _dofs0))
  {
    assert(g);
    assert(V);
    if (g->shape.size() != V->element()->value_shape().size())
    {
      throw std::runtime_error(
          "Rank mis-match between Constant and function space in DirichletBC");
    }

    if (g->value.size() != _function_space->dofmap()->bs())
    {
      throw std::runtime_error(
          "Creating a DirichletBC using a Constant is not supported when the "
          "Constant size is not equal to the block size of the constrained "
          "(sub-)space. Use a fem::Function to create the fem::DirichletBC.");
    }

    if (!V->element()->interpolation_ident())
    {
      throw std::runtime_error(
          "Constant can be used only with point-evaluation elements");
    }

    // Unroll _dofs0 if dofmap block size > 1
    if (const int bs = V->dofmap()->bs(); bs > 1)
    {
      _owned_indices0 *= bs;
      _dofs0 = unroll_dofs(_dofs0, bs);
    }
  }

  /// @brief Create a representation of a Dirichlet boundary condition
  /// where the space being constrained is the same as the function that
  /// defines the constraint Function, i.e. share the same
  /// fem::FunctionSpace.
  ///
  /// @pre `dofs` must be sorted.
  ///
  /// @param[in] g The boundary condition value.
  /// @param[in] dofs Degree-of-freedom block indices to be constrained.
  /// @note The indices in `dofs` are for *blocks*, e.g. a block index
  /// corresponds to 3 degrees-of-freedom if the dofmap associated with
  /// `g` has block size 3.
  template <std::convertible_to<std::vector<std::int32_t>> X>
  DirichletBC(std::shared_ptr<const Function<T, U>> g, X&& dofs)
      : _function_space(g->function_space()), _g(g),
        _dofs0(std::forward<X>(dofs)),
        _owned_indices0(num_owned(*_function_space->dofmap(), _dofs0))
  {
    assert(_function_space);

    // Unroll _dofs0 if dofmap block size > 1
    if (const int bs = _function_space->dofmap()->bs(); bs > 1)
    {
      _owned_indices0 *= bs;
      _dofs0 = unroll_dofs(_dofs0, bs);
    }
  }

  /// @brief Create a representation of a Dirichlet boundary condition
  /// where the space being constrained and the function that defines
  /// the constraint values do not share the same fem::FunctionSpace.
  ///
  /// A typical example is when applying a constraint on a subspace. The
  /// (sub)space and the constrain function must have the same finite
  /// element.
  ///
  /// @pre The two degree-of-freedom arrays in `V_g_dofs` must be
  /// sorted by the indices in the first array.
  ///
  /// @param[in] g The boundary condition value
  /// @param[in] V_g_dofs Two arrays of degree-of-freedom indices
  /// (`std::array<std::vector<std::int32_t>, 2>`). First array are
  /// indices in the space where boundary condition is applied (V),
  /// second array are indices in the space of the boundary condition
  /// value function g. The dof indices are unrolled, i.e. are not by
  /// dof block.
  /// @param[in] V The function (sub)space on which the boundary
  /// condition is applied
  /// @note The indices in `dofs` are unrolled and not for blocks.
  template <typename X>
  DirichletBC(std::shared_ptr<const Function<T, U>> g, X&& V_g_dofs,
              std::shared_ptr<const FunctionSpace<U>> V)
      : _function_space(V), _g(g),
        _dofs0(std::forward<typename X::value_type>(V_g_dofs[0])),
        _dofs1_g(std::forward<typename X::value_type>(V_g_dofs[1])),
        _owned_indices0(num_owned(*_function_space->dofmap(), _dofs0))
  {
  }

  /// Copy constructor
  /// @param[in] bc The object to be copied
  DirichletBC(const DirichletBC& bc) = default;

  /// Move constructor
  /// @param[in] bc The object to be moved
  DirichletBC(DirichletBC&& bc) = default;

  /// Destructor
  ~DirichletBC() = default;

  /// Assignment operator
  /// @param[in] bc Another DirichletBC object
  DirichletBC& operator=(const DirichletBC& bc) = default;

  /// Move assignment operator
  DirichletBC& operator=(DirichletBC&& bc) = default;

  /// The function space to which boundary conditions are applied
  /// @return The function space
  std::shared_ptr<const FunctionSpace<U>> function_space() const
  {
    return _function_space;
  }

  /// Return boundary value function g
  /// @return The boundary values Function
  std::variant<std::shared_ptr<const Function<T, U>>,
               std::shared_ptr<const Constant<T>>>
  value() const
  {
    return _g;
  }

  /// Access dof indices (local indices, unrolled), including ghosts, to
  /// which a Dirichlet condition is applied, and the index to the first
  /// non-owned (ghost) index. The array of indices is sorted.
  /// @return Sorted array of dof indices (unrolled) and index to the
  /// first entry in the dof index array that is not owned. Entries
  /// `dofs[:pos]` are owned and entries `dofs[pos:]` are ghosts.
  std::pair<std::span<const std::int32_t>, std::int32_t> dof_indices() const
  {
    return {_dofs0, _owned_indices0};
  }

  /// Set bc entries in `x` to `scale * x_bc`
  ///
  /// @param[in] x The array in which to set `scale * x_bc[i]`, where
  /// x_bc[i] is the boundary value of x[i]. Entries in x that do not
  /// have a Dirichlet condition applied to them are unchanged. The
  /// length of x must be less than or equal to the index of the
  /// greatest boundary dof index. To set values only for
  /// degrees-of-freedom that are owned by the calling rank, the length
  /// of the array @p x should be equal to the number of dofs owned by
  /// this rank.
  /// @param[in] scale The scaling value to apply
  void set(std::span<T> x, T scale = 1) const
  {
    if (std::holds_alternative<std::shared_ptr<const Function<T, U>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Function<T, U>>>(_g);
      assert(g);
      std::span<const T> values = g->x()->array();
      auto dofs1_g = _dofs1_g.empty() ? std::span(_dofs0) : std::span(_dofs1_g);
      std::int32_t x_size = x.size();
      for (std::size_t i = 0; i < _dofs0.size(); ++i)
      {
        if (_dofs0[i] < x_size)
        {
          assert(dofs1_g[i] < (std::int32_t)values.size());
          x[_dofs0[i]] = scale * values[dofs1_g[i]];
        }
      }
    }
    else if (std::holds_alternative<std::shared_ptr<const Constant<T>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Constant<T>>>(_g);
      std::vector<T> value = g->value;
      int bs = _function_space->dofmap()->bs();
      std::int32_t x_size = x.size();
      std::for_each(_dofs0.cbegin(), _dofs0.cend(),
                    [x_size, bs, scale, &value, &x](auto dof)
                    {
                      if (dof < x_size)
                        x[dof] = scale * value[dof % bs];
                    });
    }
  }

  /// Set bc entries in `x` to `scale * (x0 - x_bc)`
  /// @param[in] x The array in which to set `scale * (x0 - x_bc)`
  /// @param[in] x0 The array used in compute the value to set
  /// @param[in] scale The scaling value to apply
  void set(std::span<T> x, std::span<const T> x0, T scale = 1) const
  {
    if (std::holds_alternative<std::shared_ptr<const Function<T, U>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Function<T, U>>>(_g);
      assert(g);
      std::span<const T> values = g->x()->array();
      assert(x.size() <= x0.size());
      auto dofs1_g = _dofs1_g.empty() ? std::span(_dofs0) : std::span(_dofs1_g);
      std::int32_t x_size = x.size();
      for (std::size_t i = 0; i < _dofs0.size(); ++i)
      {
        if (_dofs0[i] < x_size)
        {
          assert(dofs1_g[i] < (std::int32_t)values.size());
          x[_dofs0[i]] = scale * (values[dofs1_g[i]] - x0[_dofs0[i]]);
        }
      }
    }
    else if (std::holds_alternative<std::shared_ptr<const Constant<T>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Constant<T>>>(_g);
      const std::vector<T>& value = g->value;
      std::int32_t bs = _function_space->dofmap()->bs();
      std::for_each(_dofs0.begin(), _dofs0.end(),
                    [&x, &x0, &value, scale, bs](auto dof)
                    {
                      if (dof < (std::int32_t)x.size())
                        x[dof] = scale * (value[dof % bs] - x0[dof]);
                    });
    }
  }

  /// @todo Review this function - it is almost identical to the
  /// 'DirichletBC::set' function
  ///
  /// Set boundary condition value for entries with an applied boundary
  /// condition. Other entries are not modified.
  /// @param[out] values The array in which to set the dof values.
  /// The array must be at least as long as the array associated with V1
  /// (the space of the function that provides the dof values)
  void dof_values(std::span<T> values) const
  {
    if (std::holds_alternative<std::shared_ptr<const Function<T, U>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Function<T, U>>>(_g);
      assert(g);
      std::span<const T> g_values = g->x()->array();
      auto dofs1_g = _dofs1_g.empty() ? std::span(_dofs0) : std::span(_dofs1_g);
      for (std::size_t i = 0; i < dofs1_g.size(); ++i)
        values[_dofs0[i]] = g_values[dofs1_g[i]];
    }
    else if (std::holds_alternative<std::shared_ptr<const Constant<T>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Constant<T>>>(_g);
      assert(g);
      const std::vector<T>& g_value = g->value;
      const std::int32_t bs = _function_space->dofmap()->bs();
      for (std::size_t i = 0; i < _dofs0.size(); ++i)
        values[_dofs0[i]] = g_value[_dofs0[i] % bs];
    }
  }

  /// Set markers[i] = true if dof i has a boundary condition applied.
  /// Value of markers[i] is not changed otherwise.
  /// @param[in,out] markers Entry makers[i] is set to true if dof i in
  /// V0 had a boundary condition applied, i.e. dofs which are fixed by
  /// a boundary condition. Other entries in @p markers are left
  /// unchanged.
  void mark_dofs(std::span<std::int8_t> markers) const
  {
    for (std::size_t i = 0; i < _dofs0.size(); ++i)
    {
      assert(_dofs0[i] < (std::int32_t)markers.size());
      markers[_dofs0[i]] = true;
    }
  }

private:
  // The function space (possibly a sub function space)
  std::shared_ptr<const FunctionSpace<U>> _function_space;

  // The function
  std::variant<std::shared_ptr<const Function<T, U>>,
               std::shared_ptr<const Constant<T>>>
      _g;

  // Dof indices (_dofs0) in _function_space and (_dofs1_g) in the
  // space of _g. _dofs1_g may be empty if _dofs0 can be re-used
  std::vector<std::int32_t> _dofs0, _dofs1_g;

  // The first _owned_indices in _dofs are owned by this process
  std::int32_t _owned_indices0 = -1;
};
} // namespace dolfinx::fem
