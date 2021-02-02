// Copyright (C) 2007-2020 Michal Habera, Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Core>
#include <array>
#include <dolfinx/common/span.hpp>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/utils.h>
#include <functional>
#include <memory>
#include <vector>

namespace dolfinx
{

namespace mesh
{
class Mesh;
} // namespace mesh

namespace fem
{

/// Find degrees-of-freedom which belong to the provided mesh entities
/// (topological). Note that degrees-of-freedom for discontinuous
/// elements are associated with the cell even if they may appear to be
/// associated with a facet/edge/vertex.
///
/// @param[in] V The function (sub)spaces on which degrees-of-freedom
///   (DOFs) will be located. The spaces must share the same mesh and
///   element type.
/// @param[in] dim Topological dimension of mesh entities on which
///   degrees-of-freedom will be located
/// @param[in] entities Indices of mesh entities. All DOFs associated
///   with the closure of these indices will be returned
/// @param[in] remote True to return also "remotely located"
///   degree-of-freedom indices. Remotely located degree-of-freedom
///   indices are local/owned by the current process, but which the
///   current process cannot identify because it does not recognize mesh
///   entity as a marked. For example, a boundary condition dof at a
///   vertex where this process does not have the associated boundary
///   facet. This commonly occurs with partitioned meshes.
/// @return Array of DOF indices (local to the MPI rank) in the spaces
/// V[0] and V[1]. The array[0](i) entry is the DOF index in the space
/// V[0] and array[1](i) is the correspinding DOF entry in the space
/// V[1]. The returned dofs are 'unrolled', i.e. block size = 1.
std::array<std::vector<std::int32_t>, 2> locate_dofs_topological(
    const std::array<std::reference_wrapper<const fem::FunctionSpace>, 2>& V,
    const int dim, const tcb::span<const std::int32_t>& entities,
    bool remote = true);

/// Find degrees-of-freedom which belong to the provided mesh entities
/// (topological). Note that degrees-of-freedom for discontinuous
/// elements are associated with the cell even if they may appear to be
/// associated with a facet/edge/vertex.
///
/// @param[in] V The function (sub)space on which degrees-of-freedom
///   (DOFs) will be located.
/// @param[in] dim Topological dimension of mesh entities on which
///   degrees-of-freedom will be located
/// @param[in] entities Indices of mesh entities. All DOFs associated
///   with the closure of these indices will be returned
/// @param[in] remote True to return also "remotely located"
///   degree-of-freedom indices. Remotely located degree-of-freedom
///   indices are local/owned by the current process, but which the
///   current process cannot identify because it does not recognize mesh
///   entity as a marked. For example, a boundary condition dof at a
///   vertex where this process does not have the associated boundary
///   facet. This commonly occurs with partitioned meshes.
/// @return Array of DOF index blocks (local to the MPI rank) in the
/// space V. The array uses the block size of the dofmap associated
/// with V.
std::vector<std::int32_t>
locate_dofs_topological(const fem::FunctionSpace& V, const int dim,
                        const tcb::span<const std::int32_t>& entities,
                        bool remote = true);

/// Finds degrees of freedom whose geometric coordinate is true for the
/// provided marking function.
///
/// @attention This function is slower than the topological version
///
/// @param[in] V The function (sub)space(s) on which degrees of freedom
///     will be located. The spaces must share the same mesh and
///     element type.
/// @param[in] marker_fn Function marking tabulated degrees of freedom
/// @return Array of DOF indices (local to the MPI rank) in the spaces
/// V[0] and V[1]. The array[0](i) entry is the DOF index in the space
/// V[0] and array[1](i) is the correspinding DOF entry in the space
/// V[1]. The returned dofs are 'unrolled', i.e. block size = 1.
std::array<std::vector<std::int32_t>, 2> locate_dofs_geometrical(
    const std::array<std::reference_wrapper<const fem::FunctionSpace>, 2>& V,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker_fn);

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
    const fem::FunctionSpace& V,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker_fn);

/// Interface for setting (strong) Dirichlet boundary conditions
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

public:
  /// Create a representation of a Dirichlet boundary condition where
  /// the space being constrained is the same as the function that
  /// defines the constraint values, i.e. share the same
  /// `FunctionSpace`.
  ///
  /// @param[in] g The boundary condition value. The boundary condition
  /// can be applied to a a function on the same space as g.
  /// @param[in] dofs Degree-of-freedom block indices (@p
  /// std::vector<std::int32_t>) in the space of the boundary value
  /// function applied to V_dofs[i]. The dof block indices must be
  /// sorted.
  /// @note The indices in `dofs` are for *blocks*, e.g. a block index
  /// maps to 3 degrees-of-freedom if the dofmap associated with `g` has
  /// block size 3
  template <typename U>
  DirichletBC(const std::shared_ptr<const fem::Function<T>>& g, U&& dofs)
      : _function_space(g->function_space()), _g(g),
        _dofs0(std::forward<U>(dofs))
  {
    const int owned_size0 = _function_space->dofmap()->index_map->size_local();
    auto it = std::lower_bound(_dofs0.begin(), _dofs0.end(), owned_size0);
    const int map0_bs = _function_space->dofmap()->index_map_bs();
    _owned_indices0 = map0_bs * std::distance(_dofs0.begin(), it);

    const int bs = _function_space->dofmap()->bs();
    if (bs > 1)
    {
      // Unroll for the block size
      const std::vector<std::int32_t> dof_tmp = _dofs0;
      _dofs0.resize(bs * dof_tmp.size());
      for (std::size_t i = 0; i < dof_tmp.size(); ++i)
      {
        for (int k = 0; k < bs; ++k)
          _dofs0[bs * i + k] = bs * dof_tmp[i] + k;
      }
    }

    // TODO: allows single dofs array (let one point to the other)
    _dofs1_g = _dofs0;
  }

  /// Create a representation of a Dirichlet boundary condition where
  /// the space being constrained and the function that defines the
  /// constraint values do not share the same `FunctionSpace`. A typical
  /// examples is when applying a constraint on a subspace. The
  /// (sub)space and the constrain function must have the same finite
  /// element.
  ///
  /// @param[in] g The boundary condition value
  /// @param[in] V_g_dofs Two arrays of degree-of-freedom indices. First
  /// array are indices in the space where boundary condition is applied
  /// (V), second array are indices in the space of the boundary
  /// condition value function g. The arrays must be sorted by the
  /// indices in the first array. The dof indices are unrolled, i.e. are
  /// not by dof block.
  /// @param[in] V The function (sub)space on which the boundary
  /// condition is applied
  /// @note The indices in `dofs` are unrolled and not for blocks
  DirichletBC(const std::shared_ptr<const fem::Function<T>>& g,
              const std::array<std::vector<std::int32_t>, 2>& V_g_dofs,
              std::shared_ptr<const fem::FunctionSpace> V)
      : _function_space(V), _g(g), _dofs0(V_g_dofs[0]), _dofs1_g(V_g_dofs[1])
  {
    assert(_dofs0.size() == _dofs1_g.size());
    assert(_function_space);
    assert(_g);

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
  std::shared_ptr<const fem::Function<T>> value() const { return _g; }

  /// Access dof indices (local indices, unrolled), including ghosts, to
  /// which a Dirichlet condition is applied, and the index to the first
  /// non-owned (ghost) index. The array of indices is sorted.
  /// @return Sorted array of dof indices (unrolled) and index to the
  /// first entry in the dof index array that is not owned. Entries
  /// `dofs[:pos]` are owned and entries `dofs[pos:]` are ghosts.
  std::pair<tcb::span<const std::int32_t>, std::int32_t> dof_indices() const
  {
    return {tcb::make_span(_dofs0), _owned_indices0};
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
  void set(tcb::span<T> x, double scale = 1.0) const
  {
    assert(_g);
    const std::vector<T>& g = _g->x()->array();
    for (std::size_t i = 0; i < _dofs0.size(); ++i)
    {
      if (_dofs0[i] < (std::int32_t)x.size())
      {
        assert(_dofs1_g[i] < (std::int32_t)g.size());
        x[_dofs0[i]] = scale * g[_dofs1_g[i]];
      }
    }
  }

  /// Set bc entries in `x` to `scale * (x0 - x_bc)`
  /// @param[in] x The array in which to set `scale * (x0 - x_bc)`
  /// @param[in] x0 The array used in compute the value to set
  /// @param[in] scale The scaling value to apply
  void set(tcb::span<T> x, const tcb::span<const T>& x0,
           double scale = 1.0) const
  {
    assert(_g);
    const std::vector<T>& g = _g->x()->array();
    assert(x.size() <= x0.size());
    for (std::size_t i = 0; i < _dofs0.size(); ++i)
    {
      if (_dofs0[i] < (std::int32_t)x.size())
      {
        assert(_dofs1_g[i] < (std::int32_t)g.size());
        x[_dofs0[i]] = scale * (g[_dofs1_g[i]] - x0[_dofs0[i]]);
      }
    }
  }

  /// @todo Review this function - it is almost identical to the
  /// 'DirichletBC::set' function
  ///
  /// Set boundary condition value for entries with an applied boundary
  /// condition. Other entries are not modified.
  /// @param[in,out] values The array in which to set the dof values.
  /// The array must be at least as long as the array associated with V1
  /// (the space of the function that provides the dof values)
  void dof_values(tcb::span<T> values) const
  {
    assert(_g);
    const std::vector<T>& g = _g->x()->array();
    for (std::size_t i = 0; i < _dofs1_g.size(); ++i)
      values[_dofs0[i]] = g[_dofs1_g[i]];
  }

  /// Set markers[i] = true if dof i has a boundary condition applied.
  /// Value of markers[i] is not changed otherwise.
  /// @param[in,out] markers Entry makers[i] is set to true if dof i in
  /// V0 had a boundary condition applied, i.e. dofs which are fixed by
  /// a boundary condition. Other entries in @p markers are left
  /// unchanged.
  void mark_dofs(std::vector<bool>& markers) const
  {
    for (std::size_t i = 0; i < _dofs0.size(); ++i)
    {
      assert(_dofs0[i] < (std::int32_t)markers.size());
      markers[_dofs0[i]] = true;
    }
  }

private:
  // The function space (possibly a sub function space)
  std::shared_ptr<const fem::FunctionSpace> _function_space;

  // The function
  std::shared_ptr<const fem::Function<T>> _g;

  // Dof indices (_dofs0) in _function_space and (_dofs1_g) in the
  // space of _g
  std::vector<std::int32_t> _dofs0, _dofs1_g;

  // The first _owned_indices in _dofs are owned by this process
  int _owned_indices0 = -1;
  int _owned_indices1 = -1;
};
} // namespace fem
} // namespace dolfinx
