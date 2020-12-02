// Copyright (C) 2007-2020 Michal Habera, Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <dolfinx/function/Function.h>
#include <dolfinx/la/utils.h>
#include <functional>
#include <memory>
#include <vector>

namespace dolfinx
{

namespace function
{
template <typename T>
class Function;
class FunctionSpace;
} // namespace function

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
/// @return Array of local DOF indices in the spaces V[0] and V[1]. The
///   array[0](i) entry is the DOF index in the space V[0] and array[1](i)
///   is the correspinding DOF entry in the space V[1].
std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>
locate_dofs_topological(
    const std::array<std::reference_wrapper<const function::FunctionSpace>, 2>&
        V,
    const int dim, const Eigen::Ref<const Eigen::ArrayXi>& entities,
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
/// @return Array of local DOF indices in the spaces .
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
locate_dofs_topological(const function::FunctionSpace& V, const int dim,
                        const Eigen::Ref<const Eigen::ArrayXi>& entities,
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
/// @return Array of local DOF indices in the spaces V[0] (and V[1] is
///     two spaces are passed in). If two spaces are passed in, the (i,
///     0) entry is the DOF index in the space V[0] and (i, 1) is the
///     correspinding DOF entry in the space V[1].
std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>
locate_dofs_geometrical(
    const std::array<std::reference_wrapper<const function::FunctionSpace>, 2>&
        V,
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
/// @return Array of local DOF indices in the spaces
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> locate_dofs_geometrical(
    const function::FunctionSpace& V,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker_fn);

/// Interface for setting (strong) Dirichlet boundary conditions
///
///     u = g on G,
///
/// where u is the solution to be computed, g is a function and G is a
/// sub domain of the mesh.
///
/// A DirichletBC is specified by the function g, the function space
/// (trial space) and degrees of freedom to which the boundary condition
/// applies.

template <typename T>
class DirichletBC
{

public:
  /// Create boundary condition
  ///
  /// @todo Comment required ordering for dofs
  ///
  /// @param[in] g The boundary condition value. The boundary condition
  /// can be applied to a a function on the same space as g.
  /// @param[in] dofs Degree-of-freedom indices in the space of the
  ///   boundary value function applied to V_dofs[i]
  DirichletBC(
      const std::shared_ptr<const function::Function<T>>& g,
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>&
          dofs)
      : _function_space(g->function_space()), _g(g), _dofs0(dofs),
        _dofs1_g(dofs)
  {
    // TODO: allows single dofs array (let one point to the other)
    const int owned_size0 = _function_space->dofmap()->index_map->size_local();
    const int map0_bs = _function_space->dofmap()->index_map_bs();

    auto* it = std::lower_bound(_dofs0.data(), _dofs0.data() + _dofs0.rows(),
                                map0_bs * owned_size0);
    _owned_indices0 = std::distance(_dofs0.data(), it);
    _owned_indices1 = _owned_indices0;
  }

  /// Create boundary condition
  ///
  /// @todo Comment required ordering for dofs
  ///
  /// @param[in] g The boundary condition value
  /// @param[in] V_g_dofs Two arrays of degree-of-freedom indices. First
  ///   array are indices in the space where boundary condition is
  ///   applied (V), second array are indices in the space of the
  ///   boundary condition value function g.
  /// @param[in] V The function (sub)space on which the boundary
  ///   condition is applied
  DirichletBC(const std::shared_ptr<const function::Function<T>>& g,
              const std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
                               2>& V_g_dofs,
              std::shared_ptr<const function::FunctionSpace> V)
      : _function_space(V), _g(g), _dofs0(V_g_dofs[0]), _dofs1_g(V_g_dofs[1])
  {
    assert(_dofs0.rows() == _dofs1_g.rows());
    assert(_function_space);
    // const int bs0 = _function_space->dofmap()->bs();
    assert(_g);
    // const int bs1 = g->function_space()->dofmap()->bs();
    // if (bs0 * _dofs0.size() != bs1 * _dofs1_g.size())
    //   throw std::runtime_error("Incompatible dof index arrays.");

    const int map0_bs = _function_space->dofmap()->index_map_bs();
    const int map0_size = _function_space->dofmap()->index_map->size_local();
    const int owned_size0 = (map0_bs * map0_size);
    auto it0 = std::lower_bound(_dofs0.data(), _dofs0.data() + _dofs0.rows(),
                                owned_size0);
    _owned_indices0 = std::distance(_dofs0.data(), it0);

    const int map1_bs = _g->function_space()->dofmap()->index_map_bs();
    const int map1_size
        = g->function_space()->dofmap()->index_map->size_local();
    const int owned_size1 = (map1_bs * map1_size);
    auto it1 = std::lower_bound(_dofs1_g.data(),
                                _dofs1_g.data() + _dofs1_g.rows(), owned_size1);
    _owned_indices1 = std::distance(_dofs1_g.data(), it1);

    // FIXME: Is this check needed?
    assert(_owned_indices0 == _owned_indices1);

    // const int owned_size1
    //     = g->function_space()->dofmap()->index_map->size_local();
    // auto it1 = std::lower_bound(_dofs1_g.data(),
    //                             _dofs1_g.data() + _dofs1_g.rows(),
    //                             owned_size1);
    // _owned_indices1 = std::distance(_dofs1_g.data(), it1);
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
  std::shared_ptr<const function::FunctionSpace> function_space() const
  {
    return _function_space;
  }

  /// Return boundary value function g
  /// @return The boundary values Function
  std::shared_ptr<const function::Function<T>> value() const { return _g; }

  /// Get array of dof indices to which a Dirichlet boundary condition
  /// is applied. The array is sorted and may contain ghost entries.
  // const std::array<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>, 2>&
  // dofs(int dim) const
  // {
  //   if (dim == 0)
  //   {
  //     const int bs = _function_space->dofmap()->bs();
  //     return {_dofs0.head(_owned_indices0), bs};
  //   }
  //   else if (dim == 1)
  //   {
  //     const int bs = _g->function_space()->dofmap()->bs();
  //     return {_dofs1_g.head(_owned_indices1), bs};
  //   }
  //   else
  //     throw std::runtime_error("Wrong dim index");
  // }

  /// Get array of dof indices owned by this process to which a
  /// Dirichlet BC is applied. The array is sorted and does not contain
  /// ghost entries.
  std::pair<
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>,
      int>
  dofs_owned(int dim) const
  {
    if (dim == 0)
    {
      const int bs = _function_space->dofmap()->bs();
      return {_dofs0.head(_owned_indices0), bs};
    }
    else if (dim == 1)
    {
      const int bs = _g->function_space()->dofmap()->bs();
      return {_dofs1_g.head(_owned_indices1), bs};
    }
    else
      throw std::runtime_error("Wrong dim index");
  }

  /// Set bc entries in x to scale*x_bc
  /// @todo Clarify w.r.t ghosts
  void set(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x,
           double scale = 1.0) const
  {
    // FIXME: This one excludes ghosts. Need to straighten out.
    assert(_g);
    // const int bs0 = _function_space->dofmap()->bs();
    // const int bs1 = _g->function_space()->dofmap()->bs();
    auto& g = _g->x()->array();
    // assert(_dofs1_g.rows() == _dofs1_g.rows());
    for (Eigen::Index i = 0; i < _dofs0.rows(); ++i)
    {
      // assert(_dofs0(i) < x.rows());
      // assert(_dofs1_g(i) < g.rows());
      if (_dofs0(i) < x.rows())
      {
        assert(_dofs1_g(i) < g.rows());
        x[_dofs0(i)] = scale * g[_dofs1_g(i)];
      }
      // for (int k0 = 0; k0 < bs0; ++k0)
      // {
      //   const std::int32_t pos = bs0 * i + k0;
      //   if (bs0 * _dofs0(i) + k0 < x.rows())
      //   {
      //     std::div_t pos1 = std::div(pos, bs1);
      //     x[bs0 * _dofs0(i) + k0]
      //         = scale * g[bs1 * _dofs1_g(pos1.quot) + pos1.rem];
      //   }
      // }
    }
  }

  /// Set bc entries in x to scale*(x0 - x_bc).
  /// @todo Clarify w.r.t ghosts
  void set(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x,
           const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x0,
           double scale = 1.0) const
  {
    // FIXME: This one excludes ghosts. Need to straighten out.
    assert(_g);
    const int bs0 = _function_space->dofmap()->bs();
    const int bs1 = _g->function_space()->dofmap()->bs();
    auto& g = _g->x()->array();
    assert(x.rows() <= x0.rows());
    for (Eigen::Index i = 0; i < _dofs0.rows(); ++i)
    {
      if (_dofs0(i) < x.rows())
      {
        assert(_dofs1_g(i) < g.rows());
        x[_dofs0(i)] = scale * (g[_dofs1_g(i)] - x0[_dofs0(i)]);
      }
      // for (int k0 = 0; k0 < bs0; ++k0)
      // {
      //   const std::int32_t pos = bs0 * i + k0;
      //   if (bs0 * _dofs0(i) + k0 < x.rows())
      //   {
      //     const std::div_t pos1 = std::div(pos, bs1);
      //     x[bs0 * _dofs0(i) + k0]
      //         = scale
      //           * (g[bs1 * _dofs1_g(pos1.quot) + pos1.rem]
      //              - x0[bs1 * _dofs0(pos1.quot) + pos1.rem]);
      //   }
      // }
    }
  }

  /// Set boundary condition value for entries with an applied boundary
  /// condition. Other entries are not modified.
  /// @todo Clarify w.r.t ghosts
  void dof_values(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> values) const
  {
    assert(_g);
    const int bs1 = _g->function_space()->dofmap()->bs();
    auto& g = _g->x()->array();
    // std::cout << "Test sizes: " << _dofs1_g.rows() << ", " << values.rows()
    //           << std::endl;
    // assert(bs1 * _dofs1_g.rows() == values.rows());
    for (Eigen::Index i = 0; i < _dofs1_g.rows(); ++i)
    {
      values[_dofs0(i)] = g[_dofs1_g(i)];
      // for (int k1 = 0; k1 < bs1; ++k1)
      // {
      //   const std::int32_t pos = bs1 * i + k1;
      //   const std::div_t pos1 = std::div(pos, bs1);
      //   assert(pos < values.rows());
      //   assert(bs1 * _dofs1_g(pos1.quot) + pos1.rem < g.rows());
      //   values[pos] = g[bs1 * _dofs1_g(pos1.quot) + pos1.rem];
      //   // values[bs0 * _dofs0(i) + k0] = g[bs1 * _dofs1_g(pos1.quot) +
      //   // pos1.rem];
      // }
    }
  }

  /// Set markers[i] = true if dof i has a boundary condition applied.
  /// Value of markers[i] is not changed otherwise.
  /// @todo Clarify w.r.t ghosts
  void mark_dofs(std::vector<bool>& markers) const
  {
    const int bs = _function_space->dofmap()->bs();
    // std::cout << "Marker bs: " << bs << std::endl;
    for (Eigen::Index i = 0; i < _dofs0.rows(); ++i)
    {
      assert(_dofs0(i) < (std::int32_t)markers.size());
      markers[_dofs0(i)] = true;
      // for (int k = 0; k < bs; ++k)
      // {
      //   assert(bs * _dofs0(i) + k < (std::int32_t)markers.size());
      //   markers[bs * _dofs0(i) + k] = true;
      //   // std::cout << "Mark true: " << bs * _dofs0(i) + k << std::endl;
      // }
    }
  }

private:
  // The function space (possibly a sub function space)
  std::shared_ptr<const function::FunctionSpace> _function_space;

  // The function
  std::shared_ptr<const function::Function<T>> _g;

  // Pairs of dof indices in _function_space (0) and in the space of _g
  // (1)
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _dofs0;
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> _dofs1_g;

  // The first _owned_indices in _dofs are owned by this process
  int _owned_indices0 = -1;
  int _owned_indices1 = -1;
}; // namespace fem
} // namespace fem
} // namespace dolfinx
