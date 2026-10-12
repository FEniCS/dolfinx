// Copyright (C) 2003-2026 Anders Logg, Garth N. Wells and Massimiliano Leoni
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include "interpolate.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dolfinx::fem
{
/// This class represents a function \f$ u_h \f$ in a finite
/// element function space \f$ V_h \f$, given by
///
/// \f[ u_h = \sum_{i=1}^{n} U_i \phi_i, \f]
/// where \f$ \{\phi_i\}_{i=1}^{n} \f$ is a basis for \f$ V_h \f$,
/// and \f$ U \f$ is a vector of expansion coefficients for \f$ u_h \f$.
///
/// @tparam T The function scalar type.
/// @tparam U The mesh geometry scalar type.
template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_t<T>>
class Function
{
public:
  /// Field type for the Function, e.g. `double`, `std::complex<float>`,
  /// etc.
  using value_type = T;
  /// Geometry type of the Mesh that the Function is defined on.
  using geometry_type = U;

  /// @brief Create function on given function space.
  /// @param[in] V The function space
  explicit Function(std::shared_ptr<const FunctionSpace<geometry_type>> V)
      : _function_space(V), _x(std::make_shared<la::Vector<value_type>>(
                                V->dofmaps().front()->index_map,
                                V->dofmaps().front()->index_map_bs()))
  {
    if (!V->component().empty())
    {
      throw std::invalid_argument(
          "Cannot create Function from subspace. Consider "
          "collapsing the function space");
    }
  }

  /// @brief Create function on given function space with a given
  /// vector.
  ///
  /// @warning This constructor is intended for internal library use
  /// only.
  ///
  /// @param[in] V The function space.
  /// @param[in] x The vector.
  Function(std::shared_ptr<const FunctionSpace<geometry_type>> V,
           std::shared_ptr<la::Vector<value_type>> x)
      : _function_space(V), _x(x)
  {
    // NOTE: We do not check for a subspace since this constructor is
    // used for creating subfunctions

    // Assertion uses '<=' to deal with sub-functions
    assert(V->dofmap());
    assert(V->dofmap()->index_map->size_global() * V->dofmap()->index_map_bs()
           <= _x->bs() * _x->index_map()->size_global());
  }

  // Copy constructor
  Function(const Function& v) = delete;

  /// Move constructor
  Function(Function&& v) = default;

  /// Destructor
  ~Function() = default;

  /// Move assignment
  Function& operator=(Function&& v) = default;

  // Assignment
  Function& operator=(const Function& v) = delete;

  /// @brief Extract a sub-function (a view into the Function).
  /// @param[in] i Index of subfunction
  /// @return The sub-function
  Function sub(int i) const
  {
    auto sub_space = std::make_shared<FunctionSpace<geometry_type>>(
        _function_space->sub({i}));
    assert(sub_space);
    return Function(sub_space, _x);
  }

  /// @brief Collapse a subfunction (view into a Function) to a
  /// stand-alone Function.
  /// @return New collapsed Function.
  Function collapse() const
  {
    // Create new collapsed FunctionSpace
    auto [V, map] = _function_space->collapse();

    // Create new vector
    auto x = std::make_shared<la::Vector<value_type>>(
        V.dofmap()->index_map, V.dofmap()->index_map_bs());

    // Copy values into new vector
    std::span<const value_type> x_old = _x->array();
    std::span<value_type> x_new = x->array();
    for (auto mapj : map)
    {
      for (std::size_t i = 0; i < mapj.size(); ++i)
      {
        assert(i < x_new.size());
        assert(static_cast<std::size_t>(mapj[i]) < x_old.size());
        x_new[i] = x_old[mapj[i]];
      }
    }

    return Function(
        std::make_shared<FunctionSpace<geometry_type>>(std::move(V)), x);
  }

  /// @brief Access the function space.
  /// @return The function space.
  std::shared_ptr<const FunctionSpace<geometry_type>> function_space() const
  {
    return _function_space;
  }

  /// @brief Underlying vector (const version).
  std::shared_ptr<const la::Vector<value_type>> x() const { return _x; }

  /// @brief Underlying vector.
  std::shared_ptr<la::Vector<value_type>> x() { return _x; }

  /// @brief Interpolate an expression f(x) over a set of cells.
  ///
  /// @param[in] f Expression function to be interpolated.
  /// @param[in] cells Cells to interpolate on.
  void interpolate(
      const std::function<
          std::pair<std::vector<value_type>, std::vector<std::size_t>>(
              md::mdspan<const geometry_type,
                         md::extents<std::size_t, 3, md::dynamic_extent>>)>& f,
      mesh::CellRange auto&& cells)
  {
    assert(_function_space);
    assert(_function_space->element());
    assert(_function_space->mesh());
    const std::vector<geometry_type> x
        = fem::interpolation_coords<geometry_type>(
            *_function_space->element(), _function_space->mesh()->geometry(),
            cells);
    md::mdspan<const geometry_type,
               md::extents<std::size_t, 3, md::dynamic_extent>>
        x_view(x.data(), 3, x.size() / 3);

    const auto [fx, fshape] = f(x_view);
    assert(fshape.size() <= 2);
    if (std::size_t vs = _function_space->element()->value_size();
        vs == 1 and fshape.size() == 1)
    {
      // Check for scalar-valued functions
      if (fshape.front() != x.size() / 3)
      {
        throw std::invalid_argument(
            "Data returned by callable has wrong length");
      }
    }
    else
    {
      // Check for vector/tensor value
      if (fshape.size() != 2)
        throw std::invalid_argument("Expected 2D array of data");

      if (fshape[0] != vs)
      {
        throw std::invalid_argument(
            "Data returned by callable has wrong shape(0) size");
      }

      if (fshape[1] != x.size() / 3)
      {
        throw std::invalid_argument(
            "Data returned by callable has wrong shape(1) size");
      }
    }

    std::array<std::size_t, 2> _fshape;
    if (fshape.size() == 1)
      _fshape = {1, fshape[0]};
    else
      _fshape = {fshape[0], fshape[1]};

    fem::interpolate(*this, std::span<const value_type>(fx.data(), fx.size()),
                     _fshape, cells);
  }

  /// @brief Interpolate an expression f(x) on the whole domain.
  ///
  /// @param[in] f Expression to be interpolated.
  void interpolate(
      const std::function<
          std::pair<std::vector<value_type>, std::vector<std::size_t>>(
              md::mdspan<const geometry_type,
                         md::extents<std::size_t, 3, md::dynamic_extent>>)>& f)
  {
    assert(_function_space);
    assert(_function_space->mesh());
    int tdim = _function_space->mesh()->topology()->dim();
    auto cmap = _function_space->mesh()->topology()->index_map(tdim);
    assert(cmap);
    std::int32_t num_cells = cmap->size_local() + cmap->num_ghosts();
    interpolate(f, std::ranges::iota_view(0, num_cells));
  }

  /// @brief Interpolate a Function over a subset of cells.
  ///
  /// The Function being interpolated from and the Function being
  /// interpolated into can be defined on different sub-meshes, i.e.
  /// views into a subset a cells.
  ///
  /// @param[in] u0 Function to be interpolated.
  /// @param[in] cells0 Cells to interpolate from. These are the indices
  /// of the cells in the mesh associated with `u0`.
  /// @param[in] cells1 Cell indices associated with the mesh of `this`
  /// that will be interpolated to. If `cells0[i]` is the index of a
  /// cell in the mesh associated with `u0`, then `cells1[i]` is the
  /// index of the *same* cell but in the mesh associated with `this`.
  ///
  /// @pre `cells0` and `cells1` must have the same length.
  void interpolate(const Function<value_type, geometry_type>& u0,
                   mesh::CellRange auto&& cells0, mesh::CellRange auto&& cells1)
  {
    fem::interpolate(*this, cells1, u0, cells0);
  }

  /// @brief Interpolate a Function over a subset of cells.
  ///
  /// The Function%s must be defined on the same mesh.
  ///
  /// @param[in] u Function to be interpolated.
  /// @param[in] cells Cells to interpolate from.
  void interpolate(const Function<value_type, geometry_type>& u,
                   mesh::CellRange auto&& cells)
  {
    fem::interpolate(*this, u, cells);
  }

  /// @brief Interpolate a Function over all cells.
  ///
  /// The Function%s must be defined on the same mesh.
  ///
  /// @param[in] u Function to be interpolated.
  void interpolate(const Function<value_type, geometry_type>& u)
  {
    assert(_function_space);
    assert(_function_space->mesh());
    int tdim = _function_space->mesh()->topology()->dim();
    auto cmap = _function_space->mesh()->topology()->index_map(tdim);
    assert(cmap);
    fem::interpolate(*this, u);
  }

  /// @brief Interpolate a Function defined on a different mesh.
  ///
  /// @param[in] u Function to be interpolated.
  /// @param[in] cells Cells in the mesh associated with `this` to
  /// interpolate into.
  /// @param[in] tol Tolerance for convergence in Newton method for non-affine
  /// pullbacks. If the mesh geometry is affine this argument is ignored.
  /// @param[in] maxit Maximum number of Newton iterations in non-affine
  /// pull-back. If the mesh geometry is affine this argument is ignored.
  /// @param[in] interpolation_data Data required for associating the
  /// interpolation points of `this` with cells in `u`. Can be computed
  /// with `fem::create_interpolation_data`.
  void interpolate(const Function<value_type, geometry_type>& u,
                   mesh::CellRange auto&& cells, double tol, int maxit,
                   const geometry::PointOwnershipData<U>& interpolation_data)
  {
    fem::interpolate(*this, u, cells, tol, maxit, interpolation_data);
  }

  /// @brief Evaluate the Function at points.
  ///
  /// @param[in] x The coordinates of the points. It has shape
  /// (num_points, 3) and storage is row-major.
  /// @param[in] xshape Shape of `x`.
  /// @param[in] cells Cell indices such that `cells[i]` is the index of
  /// the cell that contains the point `x(i)`. Negative cell indices can
  /// be passed, in which case the corresponding point is ignored.
  /// @param[out] u Values at the points. Values are not computed for
  /// points with a negative cell index. This argument must be passed
  /// with the correct size. Storage is row-major.
  /// @param[in] ushape Shape of `u`.
  /// @param[in] tol Tolerance for convergence in Newton method for non-affine
  /// pullbacks. If the mesh geometry is affine this argument is ignored.
  /// @param[in] maxit Maximum number of Newton iterations in non-affine
  /// pull-back. If the mesh geometry is affine this argument is ignored.
  void eval(std::span<const geometry_type> x, std::array<std::size_t, 2> xshape,
            mesh::CellRange auto&& cells, std::span<value_type> u,
            std::array<std::size_t, 2> ushape, double tol, int maxit) const
  {
    if (cells.empty())
      return;

    assert(x.size() == xshape[0] * xshape[1]);
    assert(u.size() == ushape[0] * ushape[1]);

    // TODO: This could be easily made more efficient by exploiting
    // points being ordered by the cell to which they belong.

    if (xshape[0] != cells.size())
    {
      throw std::invalid_argument(
          "Number of points and number of cells must be equal.");
    }

    if (xshape[0] != ushape[0])
    {
      throw std::invalid_argument(
          "Length of array for Function values must be the "
          "same as the number of points.");
    }

    // Get mesh
    assert(_function_space);
    auto mesh = _function_space->mesh();
    assert(mesh);

    // Get element
    auto element = _function_space->element();
    assert(element);
    const int bs_element = element->block_size();
    // Size of the push-forward output for one block. On a manifold a
    // Piola-mapped element has gdim physical components and only tdim
    // reference ones.
    const std::size_t value_size = element->physical_base_value_size();
    const std::size_t space_dimension = element->space_dimension() / bs_element;

    // If the space has sub elements, concatenate the evaluations on the
    // sub elements
    const int num_sub_elements = element->num_sub_elements();
    if (num_sub_elements > 1 and num_sub_elements != bs_element)
    {
      throw std::invalid_argument("Function::eval is not supported for mixed "
                                  "elements. Extract subspaces.");
    }

    // Create work vector for expansion coefficients
    std::vector<value_type> coefficients(space_dimension * bs_element);

    // Get dofmap
    std::shared_ptr<const DofMap> dofmap = _function_space->dofmap();
    assert(dofmap);
    const int bs_dof = dofmap->bs();

    std::ranges::fill(u, 0);
    std::span<const value_type> _v = _x->array();

    // Size of tensor for symmetric elements, unused in non-symmetric
    // case, but placed outside the loop for pre-computation.
    int matrix_size = 0;
    if (element->symmetric())
    {
      while (matrix_size * matrix_size < (int)ushape[1])
        ++matrix_size;
    }

    // Contract the basis at each point, pushed forward to the physical
    // cell, with the cell's expansion coefficients
    impl::evaluate_basis<geometry_type>(
        *element, *mesh, x, xshape, cells, tol, maxit,
        [&u, ushape, &coefficients, &_v, &dofmap, &element, bs_dof, bs_element,
         space_dimension, value_size,
         matrix_size](std::size_t p, std::int32_t c,
                      impl::mdspan_t<const geometry_type, 2> basis_values)
        {
          // Get degrees of freedom for current cell
          std::span<const std::int32_t> dofs = dofmap->cell_dofs(c);
          for (std::size_t i = 0; i < dofs.size(); ++i)
            for (int k = 0; k < bs_dof; ++k)
              coefficients[bs_dof * i + k] = _v[bs_dof * dofs[i] + k];

          if (element->symmetric())
          {
            // Compute expansion
            int row = 0;
            int rowstart = 0;
            for (int k = 0; k < bs_element; ++k)
            {
              if (k - rowstart > row)
              {
                row++;
                rowstart = k;
              }
              for (std::size_t i = 0; i < space_dimension; ++i)
              {
                for (std::size_t j = 0; j < value_size; ++j)
                {
                  u[p * ushape[1]
                    + (j * bs_element + row * matrix_size + k - rowstart)]
                      += coefficients[bs_element * i + k] * basis_values(i, j);
                  if (k - rowstart != row)
                  {
                    u[p * ushape[1]
                      + (j * bs_element + row + matrix_size * (k - rowstart))]
                        += coefficients[bs_element * i + k]
                           * basis_values(i, j);
                  }
                }
              }
            }
          }
          else
          {
            // Compute expansion
            for (int k = 0; k < bs_element; ++k)
            {
              for (std::size_t i = 0; i < space_dimension; ++i)
              {
                for (std::size_t j = 0; j < value_size; ++j)
                {
                  u[p * ushape[1] + (j * bs_element + k)]
                      += coefficients[bs_element * i + k] * basis_values(i, j);
                }
              }
            }
          }
        });
  }

  /// Name
  std::string name = "u";

private:
  // Function space
  std::shared_ptr<const FunctionSpace<geometry_type>> _function_space;

  // Vector of expansion coefficients (local)
  std::shared_ptr<la::Vector<value_type>> _x;
};

} // namespace dolfinx::fem
