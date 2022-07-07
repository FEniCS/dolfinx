// Copyright (C) 2007-2021 Michal Habera, Anders Logg, Garth N. Wells
// and Jørgen S.Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "Function.h"
#include "FunctionSpace.h"
#include <array>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::fem
{

/// Find degrees-of-freedom which belong to the provided mesh entities
/// (topological). Note that degrees-of-freedom for discontinuous
/// elements are associated with the cell even if they may appear to be
/// associated with a facet/edge/vertex.
///
/// @param[in] V The function (sub)space on which degrees-of-freedom
/// (DOFs) will be located.
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
locate_dofs_topological(const FunctionSpace& V, int dim,
                        const xtl::span<const std::int32_t>& entities,
                        bool remote = true);

/// Find degrees-of-freedom which belong to the provided mesh entities
/// (topological). Note that degrees-of-freedom for discontinuous
/// elements are associated with the cell even if they may appear to be
/// associated with a facet/edge/vertex.
///
/// @param[in] V The function (sub)spaces on which degrees-of-freedom
/// (DOFs) will be located. The spaces must share the same mesh and
/// element type.
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
    const std::array<std::reference_wrapper<const FunctionSpace>, 2>& V,
    int dim, const xtl::span<const std::int32_t>& entities, bool remote = true);

/// Finds degrees of freedom whose geometric coordinate is true for the
/// provided marking function.
///
/// @attention This function is slower than the topological version
///
/// @param[in] V The function (sub)space on which degrees of freedom
/// will be located.
/// @param[in] marker_fn Function marking tabulated degrees of freedom
/// @return Array of DOF index blocks (local to the MPI rank) in the
/// space V. The array uses the block size of the dofmap associated
/// with V.
std::vector<std::int32_t> locate_dofs_geometrical(
    const FunctionSpace& V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker_fn);

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
/// V[0] and array[1](i) is the correspinding DOF entry in the space
/// V[1]. The returned dofs are 'unrolled', i.e. block size = 1.
std::array<std::vector<std::int32_t>, 2> locate_dofs_geometrical(
    const std::array<std::reference_wrapper<const FunctionSpace>, 2>& V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        marker_fn);

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
template <typename T>
class DirichletBC
{
private:
  template <typename U>
  DirichletBC(const std::variant<std::shared_ptr<const Function<T>>,
                                 std::shared_ptr<const Constant<T>>>& g,
              U&& dofs, const std::shared_ptr<const FunctionSpace>& V, void*)
      : _function_space(V), _g(g), _dofs0(std::forward<U>(dofs))
  {
    assert(_function_space);
    if (auto c = std::get_if<std::shared_ptr<const Constant<T>>>(&_g))
    {
      assert(*c);
      if ((*c)->value.size() != _function_space->dofmap()->bs())
      {
        throw std::runtime_error(
            "Creating a DirichletBC using a Constant is not supported when the "
            "Constant size is not equal to the block size of the constrained "
            "(sub-)space. Use a Function to create the DirichletBC.");
      }
    }

    // Compute number of owned dofs indices in the full space (will
    // contain 'gaps' for sub-spaces)
    const int map0_bs = _function_space->dofmap()->index_map_bs();
    const int map0_size = _function_space->dofmap()->index_map->size_local();
    const int owned_size0 = map0_bs * map0_size;

    // Find number of owned indices in _dofs0
    auto it0 = std::lower_bound(_dofs0.begin(), _dofs0.end(), owned_size0);
    _owned_indices0 = std::distance(_dofs0.begin(), it0);

    // Unroll _dofs0 for dofmap block size > 1
    if (const int bs = _function_space->dofmap()->bs(); bs > 1)
    {
      _owned_indices0 *= bs;
      const std::vector<std::int32_t> dof_tmp = _dofs0;
      _dofs0.resize(bs * dof_tmp.size());
      for (std::size_t i = 0; i < dof_tmp.size(); ++i)
      {
        for (int k = 0; k < bs; ++k)
          _dofs0[bs * i + k] = bs * dof_tmp[i] + k;
      }
    }
  }

public:
  /// @brief Create a representation of a Dirichlet boundary condition
  /// constrained by a scalar or tensor constant.
  ///
  /// @param[in] g The boundary condition value (`T` or `xt::xarray<T>`)
  /// @param[in] dofs Degree-of-freedom block indices (
  /// `std::vector<std::int32_t>`) to be constrained. The indices must
  /// be sorted.
  /// @param[in] V The function space to be constrained
  /// @note Can be used only with point-evaluation elements.
  /// @note The indices in `dofs` are for *blocks*, e.g. a block index
  /// corresponds to 3 degrees-of-freedom if the dofmap associated with
  /// `g` has block size 3.
  /// @note The size of of `g` must be equal to the block size if `V`.
  /// Use the Function version if this is not the case, e.g. for some
  /// mixed spaces.
  template <typename S, typename U,
            typename = std::enable_if_t<
                std::is_convertible_v<
                    S, T> or std::is_convertible_v<S, xt::xarray<T>>>>
  DirichletBC(const S& g, U&& dofs,
              const std::shared_ptr<const FunctionSpace>& V)
      : DirichletBC(std::make_shared<Constant<T>>(g), dofs, V)
  {
  }

  /// @brief Create a representation of a Dirichlet boundary condition
  /// constrained by a fem::Constant.
  ///
  /// @param[in] g The boundary condition value
  /// @param[in] dofs Degree-of-freedom block indices (@p
  /// std::vector<std::int32_t>) to be constrained. The indices must be
  /// sorted.
  /// @param[in] V The function space to be constrained
  /// @note Can be used only with point-evaluation elements.
  /// @note The indices in `dofs` are for *blocks*, e.g. a block index
  /// corresponds to 3 degrees-of-freedom if the dofmap associated with
  /// `g` has block size 3.
  /// @note The size of of `g` must be equal to the block size if `V`.
  /// Use the Function version if this is not the case, e.g. for some
  /// mixed spaces.
  template <typename U>
  DirichletBC(const std::shared_ptr<const Constant<T>>& g, U&& dofs,
              const std::shared_ptr<const FunctionSpace>& V)
      : DirichletBC(g, dofs, V, nullptr)
  {
    assert(g);
    assert(V);
    if (V->element()->value_shape().size() != g->shape.size())
    {
      throw std::runtime_error(
          "Rank mis-match between Constant and function space in DirichletBC");
    }

    if (!V->element()->interpolation_ident())
    {
      throw std::runtime_error(
          "Constant can be used only with point-evaluation elements");
    }
  }

  /// @brief Create a representation of a Dirichlet boundary condition
  /// where the space being constrained is the same as the function that
  /// defines the constraint Function, i.e. share the same
  /// `fem::FunctionSpace`
  ///
  /// @param[in] g The boundary condition value.
  /// @param[in] dofs Degree-of-freedom block indices
  /// (`std::vector<std::int32_t>`) to be constrained. The indices must
  /// be sorted.
  /// @note The indices in `dofs` are for *blocks*, e.g. a block index
  /// corresponds to 3 degrees-of-freedom if the dofmap associated with
  /// `g` has block size 3.
  template <typename U>
  DirichletBC(const std::shared_ptr<const Function<T>>& g, U&& dofs)
      : DirichletBC(g, dofs, g->function_space(), nullptr)
  {
  }

  /// @brief Create a representation of a Dirichlet boundary condition where
  /// the space being constrained and the function that defines the
  /// constraint values do not share the same `FunctionSpace`.
  ///
  /// A typical examples is when applying a constraint on a subspace.
  /// The (sub)space and the constrain function must have the same
  /// finite element.
  ///
  /// @param[in] g The boundary condition value
  /// @param[in] V_g_dofs Two arrays of degree-of-freedom indices
  /// (`std::array<std::vector<std::int32_t>, 2>`). First array are
  /// indices in the space where boundary condition is applied (V),
  /// second array are indices in the space of the boundary condition
  /// value function g. The arrays must be sorted by the indices in the
  /// first array. The dof indices are unrolled, i.e. are not by dof
  /// block.
  /// @param[in] V The function (sub)space on which the boundary
  /// condition is applied
  /// @note The indices in `dofs` are unrolled and not for blocks.
  template <typename U>
  DirichletBC(const std::shared_ptr<const Function<T>>& g, U&& V_g_dofs,
              const std::shared_ptr<const FunctionSpace>& V)
      : _function_space(V), _g(g),
        _dofs0(std::forward<typename U::value_type>(V_g_dofs[0])),
        _dofs1_g(std::forward<typename U::value_type>(V_g_dofs[1]))
  {
    assert(_dofs0.size() == _dofs1_g.size());
    assert(_function_space);
    const int map0_bs = _function_space->dofmap()->index_map_bs();
    const int map0_size = _function_space->dofmap()->index_map->size_local();
    const int owned_size0 = map0_bs * map0_size;
    auto it0 = std::lower_bound(_dofs0.begin(), _dofs0.end(), owned_size0);
    _owned_indices0 = std::distance(_dofs0.begin(), it0);
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
  std::shared_ptr<const fem::FunctionSpace> function_space() const
  {
    return _function_space;
  }

  /// Return boundary value function g
  /// @return The boundary values Function
  std::variant<std::shared_ptr<const Function<T>>,
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
  std::pair<xtl::span<const std::int32_t>, std::int32_t> dof_indices() const
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
  void set(xtl::span<T> x, double scale = 1.0) const
  {
    if (std::holds_alternative<std::shared_ptr<const Function<T>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Function<T>>>(_g);
      assert(g);
      xtl::span<const T> values = g->x()->array();
      auto dofs1_g = _dofs1_g.empty() ? xtl::span(_dofs0) : xtl::span(_dofs1_g);
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
  void set(xtl::span<T> x, const xtl::span<const T>& x0,
           double scale = 1.0) const
  {
    if (std::holds_alternative<std::shared_ptr<const Function<T>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Function<T>>>(_g);
      assert(g);
      xtl::span<const T> values = g->x()->array();
      assert(x.size() <= x0.size());
      auto dofs1_g = _dofs1_g.empty() ? xtl::span(_dofs0) : xtl::span(_dofs1_g);
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
      std::for_each(_dofs0.cbegin(), _dofs0.cend(),
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
  void dof_values(xtl::span<T> values) const
  {
    if (std::holds_alternative<std::shared_ptr<const Function<T>>>(_g))
    {
      auto g = std::get<std::shared_ptr<const Function<T>>>(_g);
      assert(g);
      xtl::span<const T> g_values = g->x()->array();
      auto dofs1_g = _dofs1_g.empty() ? xtl::span(_dofs0) : xtl::span(_dofs1_g);
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
  void mark_dofs(const xtl::span<std::int8_t>& markers) const
  {
    for (std::size_t i = 0; i < _dofs0.size(); ++i)
    {
      assert(_dofs0[i] < (std::int32_t)markers.size());
      markers[_dofs0[i]] = true;
    }
  }

private:
  // The function space (possibly a sub function space)
  std::shared_ptr<const FunctionSpace> _function_space;

  // The function
  std::variant<std::shared_ptr<const Function<T>>,
               std::shared_ptr<const Constant<T>>>
      _g;

  // Dof indices (_dofs0) in _function_space and (_dofs1_g) in the
  // space of _g. _dofs1_g may be empty if _dofs0 can be re-used
  std::vector<std::int32_t> _dofs0, _dofs1_g;

  // The first _owned_indices in _dofs are owned by this process
  int _owned_indices0 = -1;
  int _owned_indices1 = -1;
};
} // namespace dolfinx::fem
