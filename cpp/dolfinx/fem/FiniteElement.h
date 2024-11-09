// Copyright (C) 2020-2021 Garth N. Wells and Matthew W. Scroggs
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "traits.h"
#include <array>
#include <basix/finite-element.h>
#include <concepts>
#include <cstdint>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::fem
{
/// DOF transformation type
enum class doftransform
{
  standard = 0,          ///< Standard
  transpose = 1,         ///< Transpose
  inverse = 2,           ///< Inverse
  inverse_transpose = 3, ///< Transpose inverse
};

/// @brief Basix element holder
/// @tparam T Scalar type
template <std::floating_point T>
struct BasixElementData
{
  std::reference_wrapper<const basix::FiniteElement<T>>
      element; ///< Finite element
  std::optional<std::vector<std::size_t>> value_shape
      = std::nullopt;    ///< Value shape
  bool symmetry = false; ///< symmetry
};

/// Type deduction
template <typename U, typename V, typename W>
BasixElementData(U element, V bs, W symmetry)
    -> BasixElementData<typename std::remove_cvref<U>::type::scalar_type>;

/// @brief Model of a finite element.
///
/// Provides the dof layout on a reference element, and various methods
/// for evaluating and transforming the basis.
template <std::floating_point T>
class FiniteElement
{
public:
  /// Geometry type of the Mesh that the FunctionSpace is defined on.
  using geometry_type = T;

  /// @brief Create a finite element from a Basix finite element.
  /// @param[in] element Basix finite element
  /// @param[in] block_shape The block size for the element
  /// @param[in] symmetric Is the element a symmetric tensor?
  FiniteElement(const basix::FiniteElement<geometry_type>& element,
                std::optional<std::vector<std::size_t>> block_shape
                = std::nullopt,
                bool symmetric = false);

  /// @brief Create a mixed finite element from Basix finite elements.
  /// @param[in] elements List of (Basix finite element, block size,
  /// symmetric) tuples, one for each element in the mixed element.
  FiniteElement(std::vector<BasixElementData<geometry_type>> elements);

  /// @brief Create mixed finite element from a list of finite elements.
  /// @param[in] elements Basix finite elements
  FiniteElement(
      const std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>&
          elements);

  /// @brief Create a quadrature element.
  /// @param[in] cell_type Cell type.
  /// @param[in] points Quadrature points.
  /// @param[in] pshape Shape of `points` array.
  /// @param[in] block_shape The block size for the element.
  /// @param[in] symmetric Is the element a symmetric tensor?
  FiniteElement(mesh::CellType cell_type, std::span<const geometry_type> points,
                std::array<std::size_t, 2> pshape,
                std::vector<std::size_t> block_shape = {},
                bool symmetric = false);

  /// Copy constructor
  FiniteElement(const FiniteElement& element) = delete;

  /// Move constructor
  FiniteElement(FiniteElement&& element) = default;

  /// Destructor
  ~FiniteElement() = default;

  /// Copy assignment
  FiniteElement& operator=(const FiniteElement& element) = delete;

  /// Move assignment
  FiniteElement& operator=(FiniteElement&& element) = default;

  /// Check if two elements are equivalent
  /// @return True is the two elements are the same
  /// @note Equality can be checked only for non-mixed elements. For a
  /// mixed element, this function will raise an exception.
  bool operator==(const FiniteElement& e) const;

  /// Check if two elements are not equivalent
  /// @return True is the two elements are not the same
  /// @note Equality can be checked only for non-mixed elements. For a
  /// mixed element, this function will raise an exception.
  bool operator!=(const FiniteElement& e) const;

  /// @brief Cell shape
  mesh::CellType cell_type() const noexcept;

  /// @brief String identifying the finite element.
  /// @return Element signature
  /// @warning The function is provided for convenience, but it should
  /// not be relied upon for determining the element type. Use other
  /// functions, commonly returning enums, to determine element
  /// properties.
  std::string signature() const noexcept;

  /// Dimension of the finite element function space (the number of
  /// degrees-of-freedom for the element)
  /// @return Dimension of the finite element space
  int space_dimension() const noexcept;

  /// @brief Block size of the finite element function space.
  ///
  /// For BlockedElements, this is the number of DOFs colocated at each
  /// DOF point. For other elements, this is always 1.
  /// @return Block size of the finite element space
  int block_size() const noexcept;

  /// @brief Value size (new).
  ///
  /// The value size is the product of the value shape, e.g. is is  1
  /// for a scalar function, 2 for a 2D vector, 9 for a second-order
  /// tensor in 3D.
  /// @throws Exception is thrown for a mixed element as mixed elements
  /// do not have a value shape.
  /// @return The value size.
  int value_size() const;

  /// @brief Value shape (new, ).
  ///
  /// The value shape described the shape of the finite element field,
  /// e.g. {} for a scalar, {3, 3} for a tensor in 3D. Mixed elements do
  /// not have a value shape.
  /// @throws Exception is thrown for a mixed element as mixed elements
  /// do not have a value shape.
  /// @return The value shape.
  std::span<const std::size_t> value_shape() const;

  /// @brief Value size.
  ///
  /// The value size is the product of the value shape, e.g. is is  1
  /// for a scalar function, 2 for a 2D vector, 9 for a second-order
  /// tensor in 3D.
  /// @throws Exception is thrown for a mixed element as mixed elements
  /// do not have a value shape.
  /// @return The value size.
  int reference_value_size() const;

  /// @brief Value shape.
  ///
  /// The value shape described the shape of the finite element field,
  /// e.g. {} for a scalar, {3, 3} for a tensor in 3D. Mixed elements do
  /// not have a value shape.
  /// @throws Exception is thrown for a mixed element as mixed elements
  /// do not have a value shape.
  /// @return The value shape.
  std::span<const std::size_t> reference_value_shape() const;

  /// The local DOFs associated with each subentity of the cell
  const std::vector<std::vector<std::vector<int>>>&
  entity_dofs() const noexcept;

  /// The local DOFs associated with the closure of each subentity of the cell
  const std::vector<std::vector<std::vector<int>>>&
  entity_closure_dofs() const noexcept;

  /// Does the element represent a symmetric 2-tensor?
  bool symmetric() const;

  /// @brief Evaluate derivatives of the basis functions up to given order
  /// at points in the reference cell.
  /// @param[in,out] values Array that will be filled with the tabulated
  /// basis values. Must have shape `(num_derivatives, num_points,
  /// num_dofs, reference_value_size)` (row-major storage)
  /// @param[in] X The reference coordinates at which to evaluate the
  /// basis functions. Shape is `(num_points, topological dimension)`
  /// (row-major storage)
  /// @param[in] shape The shape of `X`
  /// @param[in] order The number of derivatives (up to and including
  /// this order) to tabulate for
  void tabulate(std::span<geometry_type> values,
                std::span<const geometry_type> X,
                std::array<std::size_t, 2> shape, int order) const;

  /// Evaluate all derivatives of the basis functions up to given order
  /// at given points in reference cell
  /// @param[in] X The reference coordinates at which to evaluate the
  /// basis functions. Shape is `(num_points, topological dimension)`
  /// (row-major storage)
  /// @param[in] shape The shape of `X`
  /// @param[in] order The number of derivatives (up to and including
  /// this order) to tabulate for
  /// @return Basis function values and array shape (row-major storage)
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 4>>
  tabulate(std::span<const geometry_type> X, std::array<std::size_t, 2> shape,
           int order) const;

  /// @brief Number of sub elements (for a mixed or blocked element).
  /// @return The number of sub elements
  int num_sub_elements() const noexcept;

  /// @brief Check if element is a mixed element.
  ///
  /// A mixed element i composed of two or more elements of different
  /// types (a block element, e.g. a Lagrange element with block size >=
  /// 1 is not considered mixed).
  ///
  /// @return True if element is mixed.
  bool is_mixed() const noexcept;

  /// Get subelements (if any)
  const std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>&
  sub_elements() const noexcept;

  /// Extract sub finite element for component
  std::shared_ptr<const FiniteElement<geometry_type>>
  extract_sub_element(const std::vector<int>& component) const;

  /// @brief Return underlying Basix element (if it exists).
  /// @throws Throws a std::runtime_error is there no Basix element.
  const basix::FiniteElement<geometry_type>& basix_element() const;

  /// @brief Get the map type used by the element
  basix::maps::type map_type() const;

  /// Check if interpolation into the finite element space is an
  /// identity operation given the evaluation on an expression at
  /// specific points, i.e. the degree-of-freedom are equal to point
  /// evaluations. The function will return `true` for Lagrange
  /// elements.
  /// @return True if interpolation is an identity operation
  bool interpolation_ident() const noexcept;

  /// Check if the push forward/pull back map from the values on
  /// reference to the values on a physical cell for this element is the
  /// identity map.
  /// @return True if the map is the identity
  bool map_ident() const noexcept;

  /// @brief Points on the reference cell at which an expression needs
  /// to be evaluated in order to interpolate the expression in the
  /// finite element space.
  ///
  /// For Lagrange elements the points will just be the nodal positions.
  /// For other elements the points will typically be the quadrature
  /// points used to evaluate moment degrees of freedom.
  /// @return Interpolation point coordinates on the reference cell,
  /// returning the (0) coordinates data (row-major) storage and (1) the
  /// shape `(num_points, tdim)`.
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>>
  interpolation_points() const;

  /// Interpolation operator (matrix) `Pi` that maps a function
  /// evaluated at the points provided by
  /// FiniteElement::interpolation_points to the element degrees of
  /// freedom, i.e. dofs = Pi f_x. See the Basix documentation for
  /// basix::FiniteElement::interpolation_matrix for how the data in
  /// `f_x` should be ordered.
  /// @return The interpolation operator `Pi`, returning the data for
  /// `Pi` (row-major storage) and the shape `(num_dofs, num_points *
  /// value_size)`
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>>
  interpolation_operator() const;

  /// @brief Create a matrix that maps degrees of freedom from one
  /// element to this element (interpolation).
  ///
  /// @param[in] from The element to interpolate from
  /// @return Matrix operator that maps the `from` degrees-of-freedom to
  /// the degrees-of-freedom of this element. The (0) matrix data
  /// (row-major storage) and (1) the shape (num_dofs of `this` element,
  /// num_dofs of `from`) are returned.
  ///
  /// @pre The two elements must use the same mapping between the
  /// reference and physical cells
  /// @note Does not support mixed elements
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>>
  create_interpolation_operator(const FiniteElement& from) const;

  /// @brief Check if DOF transformations are needed for this element.
  ///
  /// DOF transformations will be needed for elements which might not be
  /// continuous when two neighbouring cells disagree on the orientation
  /// of a shared sub-entity, and when this cannot be corrected for by
  /// permuting the DOF numbering in the dofmap.
  ///
  /// For example, Raviart-Thomas elements will need DOF
  /// transformations, as the neighbouring cells may disagree on the
  /// orientation of a basis function, and this orientation cannot be
  /// corrected for by permuting the DOF numbers on each cell.
  ///
  /// @return True if DOF transformations are required
  bool needs_dof_transformations() const noexcept;

  /// @brief Check if DOF permutations are needed for this element.
  ///
  /// DOF permutations will be needed for elements which might not be
  /// continuous when two neighbouring cells disagree on the orientation
  /// of a shared subentity, and when this can be corrected for by
  /// permuting the DOF numbering in the dofmap.
  ///
  /// For example, higher order Lagrange elements will need DOF
  /// permutations, as the arrangement of DOFs on a shared sub-entity
  /// may be different from the point of view of neighbouring cells, and
  /// this can be corrected for by permuting the DOF numbers on each
  /// cell.
  ///
  /// @return True if DOF transformations are required
  bool needs_dof_permutations() const noexcept;

  /// @brief Return a function that applies a DOF transformation
  /// operator to some data (see T_apply()).
  ///
  /// The transformation is applied from the left-hand side, i.e.
  /// \f[ u \leftarrow T u. \f]
  ///
  /// If the transformation for the (sub)element is a permutation only,
  /// the returned function will do change the ordering for the
  /// (sub)element as it is assumed that permutations are incorporated
  /// into the degree-of-freedom map.
  ///
  /// See the documentation for T_apply() for a description of the
  /// transformation for a single element type. This function generates
  /// a function that can apply the transformation to a mixed element.
  ///
  /// The signature of the returned function has four arguments:
  /// - [in,out] data The data to be transformed. This data is flattened
  ///   with row-major layout, shape=(num_dofs, block_size)
  /// - [in] cell_info Permutation data for the cell. The size of this
  ///   is num_cells. For elements where no transformations are required,
  ///   an empty span can be passed in.
  /// - [in] cell The cell number.
  /// - [in] n The block_size of the input data.
  ///
  /// @param[in] ttype The transformation type. Typical usage is:
  /// - doftransform::standard Transforms *basis function data* from the
  /// reference element to the conforming 'physical' element, e.g.
  /// \f$\phi = T \tilde{\phi}\f$.
  /// - doftransform::transpose Transforms *degree-of-freedom data* from
  /// the conforming (physical) ordering to the reference ordering, e.g.
  /// \f$\tilde{u} = T^{T} u\f$.
  /// - doftransform::inverse: Transforms *basis function data* from the
  /// the conforming (physical) ordering to the reference ordering, e.g.
  /// \f$\tilde{\phi} = T^{-1} \phi\f$.
  /// - doftransform::inverse_transpose: Transforms *degree-of-freedom
  /// data* from the reference element to the conforming (physical)
  /// ordering, e.g. \f$u = T^{-t} \tilde{u}\f$.
  /// @param[in] scalar_element Indicates whether the scalar
  /// transformations should be returned for a vector element
  template <typename U>
  std::function<void(std::span<U>, std::span<const std::uint32_t>, std::int32_t,
                     int)>
  dof_transformation_fn(doftransform ttype, bool scalar_element = false) const
  {
    if (!needs_dof_transformations())
    {
      // If no permutation needed, return function that does nothing
      return [](std::span<U>, std::span<const std::uint32_t>, std::int32_t, int)
      {
        // Do nothing
      };
    }

    if (!_sub_elements.empty())
    {
      if (!_reference_value_shape) // Mixed element
      {
        std::vector<std::function<void(
            std::span<U>, std::span<const std::uint32_t>, std::int32_t, int)>>
            sub_element_fns;
        std::vector<int> dims;
        for (std::size_t i = 0; i < _sub_elements.size(); ++i)
        {
          sub_element_fns.push_back(
              _sub_elements[i]->template dof_transformation_fn<U>(ttype));
          dims.push_back(_sub_elements[i]->space_dimension());
        }

        return [dims, sub_element_fns](std::span<U> data,
                                       std::span<const std::uint32_t> cell_info,
                                       std::int32_t cell, int block_size)
        {
          std::size_t offset = 0;
          for (std::size_t e = 0; e < sub_element_fns.size(); ++e)
          {
            const std::size_t width = dims[e] * block_size;
            sub_element_fns[e](data.subspan(offset, width), cell_info, cell,
                               block_size);
            offset += width;
          }
        };
      }
      else if (!scalar_element)
      {
        // Blocked element
        std::function<void(std::span<U>, std::span<const std::uint32_t>,
                           std::int32_t, int)>
            sub_fn = _sub_elements[0]->template dof_transformation_fn<U>(ttype);
        const int ebs = _bs;
        return [ebs, sub_fn](std::span<U> data,
                             std::span<const std::uint32_t> cell_info,
                             std::int32_t cell, int data_block_size)
        { sub_fn(data, cell_info, cell, ebs * data_block_size); };
      }
    }

    switch (ttype)
    {
    case doftransform::inverse_transpose:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      { Tt_inv_apply(data, cell_info[cell], block_size); };
    case doftransform::transpose:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      { Tt_apply(data, cell_info[cell], block_size); };
    case doftransform::inverse:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      { Tinv_apply(data, cell_info[cell], block_size); };
    case doftransform::standard:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      { T_apply(data, cell_info[cell], block_size); };
    default:
      throw std::runtime_error("Unknown transformation type");
    }
  }

  /// @brief Return a function that applies DOF transformation to some
  /// transposed data (see T_apply_right()).
  ///
  /// The transformation is applied from the right-hand side, i.e.
  /// \f[ u^{t} \leftarrow u^{t} T. \f]
  ///
  /// If the transformation for the (sub)element is a permutation only,
  /// the returned function will do change the ordering for the
  /// (sub)element as it is assumed that permutations are incorporated
  /// into the degree-of-freedom map.
  ///
  /// The signature of the returned function has four arguments:
  /// - [in,out] data The data to be transformed. This data is flattened
  ///   with row-major layout, shape=(num_dofs, block_size)
  /// - [in] cell_info Permutation data for the cell. The size of this
  ///   is num_cells. For elements where no transformations are required,
  ///   an empty span can be passed in.
  /// - [in] cell The cell number
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] ttype Transformation type. See dof_transformation_fn().
  /// @param[in] scalar_element Indicate if the scalar transformations
  /// should be returned for a vector element.
  template <typename U>
  std::function<void(std::span<U>, std::span<const std::uint32_t>, std::int32_t,
                     int)>
  dof_transformation_right_fn(doftransform ttype,
                              bool scalar_element = false) const
  {
    if (!needs_dof_transformations())
    {
      // If no permutation needed, return function that does nothing
      return [](std::span<U>, std::span<const std::uint32_t>, std::int32_t, int)
      {
        // Do nothing
      };
    }
    else if (!_sub_elements.empty())
    {
      if (!_reference_value_shape) // Mixed element
      {
        std::vector<std::function<void(
            std::span<U>, std::span<const std::uint32_t>, std::int32_t, int)>>
            sub_element_fns;
        for (std::size_t i = 0; i < _sub_elements.size(); ++i)
        {
          sub_element_fns.push_back(
              _sub_elements[i]->template dof_transformation_right_fn<U>(ttype));
        }

        return [this, sub_element_fns](std::span<U> data,
                                       std::span<const std::uint32_t> cell_info,
                                       std::int32_t cell, int block_size)
        {
          std::size_t offset = 0;
          for (std::size_t e = 0; e < sub_element_fns.size(); ++e)
          {
            sub_element_fns[e](data.subspan(offset, data.size() - offset),
                               cell_info, cell, block_size);
            offset += _sub_elements[e]->space_dimension();
          }
        };
      }
      else if (!scalar_element)
      {
        // Blocked element
        // The transformation from the left can be used here as blocked
        // elements use xyzxyzxyz ordering, and so applying the DOF
        // transformation from the right is equivalent to applying the DOF
        // transformation from the left to data using xxxyyyzzz ordering
        std::function<void(std::span<U>, std::span<const std::uint32_t>,
                           std::int32_t, int)>
            sub_fn = _sub_elements[0]->template dof_transformation_fn<U>(ttype);
        return [this, sub_fn](std::span<U> data,
                              std::span<const std::uint32_t> cell_info,
                              std::int32_t cell, int data_block_size)
        {
          const int ebs = block_size();
          const std::size_t dof_count = data.size() / data_block_size;
          for (int block = 0; block < data_block_size; ++block)
          {
            sub_fn(data.subspan(block * dof_count, dof_count), cell_info, cell,
                   ebs);
          }
        };
      }
    }

    switch (ttype)
    {
    case doftransform::inverse_transpose:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int n)
      { Tt_inv_apply_right(data, cell_info[cell], n); };
    case doftransform::transpose:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int n)
      { Tt_apply_right(data, cell_info[cell], n); };
    case doftransform::inverse:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int n)
      { Tinv_apply_right(data, cell_info[cell], n); };
    case doftransform::standard:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int n)
      { T_apply_right(data, cell_info[cell], n); };
    default:
      throw std::runtime_error("Unknown transformation type");
    }
  }

  /// @brief Transform basis functions from the reference element
  /// ordering and orientation to the globally consistent physical
  /// element ordering and orientation.
  ///
  /// Consider that the value of a finite element function \f$f_{h}\f$
  /// at a point is given by
  /// \f[
  ///  f_{h} = \phi^{T} c,
  /// \f]
  /// where \f$f_{h}\f$ has shape \f$r \times 1\f$, \f$\phi\f$ has shape
  /// \f$d \times r\f$ and holds the finite element basis functions,
  /// and \f$c\f$ has shape \f$d \times 1\f$ and holds the
  /// degrees-of-freedom. The basis functions and
  /// degree-of-freedom are with respect to the physical element
  /// orientation. If the degrees-of-freedom on the physical element
  /// orientation are given by
  /// \f[
  /// \phi = T \tilde{\phi},
  /// \f]
  /// where \f$T\f$ is a \f$d \times d\f$ matrix, it follows from
  /// \f$f_{h} = \phi^{T} c = \tilde{\phi}^{T} T^{T} c\f$ that
  /// \f[
  ///  \tilde{c} = T^{T} c.
  /// \f]
  ///
  /// This function applies \f$T\f$ to data. The transformation is
  /// performed in-place. The operator \f$T\f$ is orthogonal for many
  /// elements, but not all.
  ///
  /// This function calls the corresponding Basix function.
  ///
  /// @param[in,out] data Data to transform. The shape is `(m, n)`,
  /// where `m` is the number of dgerees-of-freedom and the storage is
  /// row-major.
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] n Number of columns in `data`.
  template <typename U>
  void T_apply(std::span<U> data, std::uint32_t cell_permutation, int n) const
  {
    assert(_element);
    _element->T_apply(data, n, cell_permutation);
  }

  /// @brief Apply the inverse transpose of the operator applied by
  /// T_apply().
  ///
  /// The transformation \f[ v = T^{-T} u \f] is performed in-place.
  ///
  /// @param[in,out] data The data to be transformed. This data is
  /// flattened with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell.
  /// @param[in] n Block_size of the input data.
  template <typename U>
  void Tt_inv_apply(std::span<U> data, std::uint32_t cell_permutation,
                    int n) const
  {
    assert(_element);
    _element->Tt_inv_apply(data, n, cell_permutation);
  }

  /// @brief Apply the transpose of the operator applied by T_apply().
  ///
  /// The transformation \f[ u \leftarrow  T^{T} u \f] is performed
  /// in-place.
  ///
  /// @param[in,out] data The data to be transformed. This data is
  /// flattened with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell.
  /// @param[in] n The block size of the input data.
  template <typename U>
  void Tt_apply(std::span<U> data, std::uint32_t cell_permutation, int n) const
  {
    assert(_element);
    _element->Tt_apply(data, n, cell_permutation);
  }

  /// @brief Apply the inverse of the operator applied by T_apply().
  ///
  /// The transformation \f[ v = T^{-1} u \f] is performed in-place.
  ///
  /// @param[in,out] data The data to be transformed. This data is
  /// flattened with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell.
  /// @param[in] n Block size of the input data.
  template <typename U>
  void Tinv_apply(std::span<U> data, std::uint32_t cell_permutation,
                  int n) const
  {
    assert(_element);
    _element->Tinv_apply(data, n, cell_permutation);
  }

  /// @brief Right(post)-apply the operator applied by T_apply().
  ///
  /// Computes \f[ v^{T} = u^{T} T \f] in-place.
  ///
  /// @param[in,out] data The data to be transformed. This data is
  /// flattened with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] n Block size of the input data
  template <typename U>
  void T_apply_right(std::span<U> data, std::uint32_t cell_permutation,
                     int n) const
  {
    assert(_element);
    _element->T_apply_right(data, n, cell_permutation);
  }

  /// @brief Right(post)-apply the inverse of the operator applied by
  /// T_apply().
  ///
  /// Computes \f[ v^{T} = u^{T} T^{-1} \f] in-place.
  ///
  /// @param[in,out] data Data to be transformed. This data is flattened
  /// with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] n Block size of the input data
  template <typename U>
  void Tinv_apply_right(std::span<U> data, std::uint32_t cell_permutation,
                        int n) const
  {
    assert(_element);
    _element->Tinv_apply_right(data, n, cell_permutation);
  }

  /// @brief Right(post)-apply the transpose of the operator applied by
  /// T_apply().
  ///
  /// Computes \f[ v^{T} = u^{T} T^{T} \f] in-place.
  ///
  /// @param[in,out] data Data to be transformed. The data is flattened
  /// with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] n Block size of the input data.
  template <typename U>
  void Tt_apply_right(std::span<U> data, std::uint32_t cell_permutation,
                      int n) const
  {
    assert(_element);
    _element->Tt_apply_right(data, n, cell_permutation);
  }

  /// @brief Right(post)-apply the transpose inverse of the operator
  /// applied by T_apply().
  ///
  /// Computes \f[ v^{T} = u^{T} T^{-T} \f] in-place.
  ///
  /// @param[in,out] data Data to be transformed. This data is flattened
  /// with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell.
  /// @param[in] n Block size of the input data.
  template <typename U>
  void Tt_inv_apply_right(std::span<U> data, std::uint32_t cell_permutation,
                          int n) const
  {
    assert(_element);
    _element->Tt_inv_apply_right(data, n, cell_permutation);
  }

  /// @brief Permute indices associated with degree-of-freedoms on the
  /// reference element ordering to the globally consistent physical
  /// element degree-of-freedom ordering.
  ///
  /// Given an array \f$\tilde{d}\f$ that holds an integer associated
  /// with each degree-of-freedom and following the reference element
  /// degree-of-freedom ordering, this function computes
  ///   \f[ d = P \tilde{d},\f]
  /// where \f$P\f$ is a permutation matrix and \f$d\f$ holds the
  /// integers in \f$\tilde{d}\f$ but permuted to follow the globally
  /// consistent physical element degree-of-freedom ordering. The
  /// permutation is computed in-place.
  ///
  /// @param[in,out] doflist Indicies associated with the
  /// degrees-of-freedom. Size=`num_dofs`.
  /// @param[in] cell_permutation Permutation data for the cell.
  void permute(std::span<std::int32_t> doflist,
               std::uint32_t cell_permutation) const;

  /// @brief Perform the inverse of the operation applied by permute().
  ///
  /// Given an array \f$d\f$ that holds an integer associated with each
  /// degree-of-freedom and following the globally consistent physical
  /// element degree-of-freedom ordering, this function computes
  /// \f[
  ///  \tilde{d} = P^{T} d,
  /// \f]
  /// where \f$P^{T}\f$ is a permutation matrix and \f$\tilde{d}\f$
  /// holds the integers in \f$d\f$ but permuted to follow the reference
  /// element degree-of-freedom ordering. The permutation is computed
  /// in-place.
  ///
  /// @param[in,out] doflist Indicies associated with the
  /// degrees-of-freedom. Size=`num_dofs`.
  /// @param[in] cell_permutation Permutation data for the cell.
  void permute_inv(std::span<std::int32_t> doflist,
                   std::uint32_t cell_permutation) const;

  /// @brief Return a function that applies a degree-of-freedom
  /// permutation to some data.
  ///
  /// The returned function can apply permute() to mixed-elements.
  ///
  /// The signature of the returned function has three arguments:
  /// - [in,out] doflist The numbers of the DOFs, a span of length num_dofs
  /// - [in] cell_permutation Permutation data for the cell
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] inverse Indicates if the inverse transformation should
  /// be returned.
  /// @param[in] scalar_element Indicates is the scalar transformations
  /// should be returned for a vector element.
  std::function<void(std::span<std::int32_t>, std::uint32_t)>
  dof_permutation_fn(bool inverse = false, bool scalar_element = false) const;

private:
  // Block size for BlockedElements. This gives the number of DOFs
  // co-located at each dof 'point'.
  std::optional<std::vector<std::size_t>> _block_shape;
  int _bs;

  mesh::CellType _cell_type;

  std::string _signature;

  int _space_dim;

  // List of sub-elements (if any)
  std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>
      _sub_elements;

  // Value space shape, e.g. {} for a scalar, {3, 3} for a tensor in 3D.
  // For a mixed element it is std::nullopt.
  std::optional<std::vector<std::size_t>> _reference_value_shape;

  // Basix Element (nullptr for mixed elements)
  std::unique_ptr<basix::FiniteElement<geometry_type>> _element;

  // Indicate whether this element represents a symmetric 2-tensor
  bool _symmetric;

  // Indicate whether the element needs permutations or transformations
  bool _needs_dof_permutations;
  bool _needs_dof_transformations;

  std::vector<std::vector<std::vector<int>>> _entity_dofs;
  std::vector<std::vector<std::vector<int>>> _entity_closure_dofs;

  // Quadrature points of a quadrature element (0 dimensional array for
  // all elements except quadrature elements)
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>> _points;
};

} // namespace dolfinx::fem
