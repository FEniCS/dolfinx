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
#include <span>
#include <utility>
#include <vector>

struct ufcx_finite_element;

namespace dolfinx::fem
{
/// Finite Element, containing the dof layout on a reference element,
/// and various methods for evaluating and transforming the basis.
template <std::floating_point T>
class FiniteElement
{
public:
  /// DOF transformation type
  enum class doftransform
  {
    standard = 0,
    transpose = 1,
    inverse = 2,
    inverse_transpose = 3,
  };

  /// Geometry type of the Mesh that the FunctionSpace is defined on.
  using geometry_type = T;

  /// @brief Create finite element from UFC finite element.
  /// @param[in] e UFC finite element.
  explicit FiniteElement(const ufcx_finite_element& e);

  /// @brief Create finite element from a Basix finite element.
  /// @param[in] element Basix finite element
  /// @param[in] block_size The block size for the element
  FiniteElement(const basix::FiniteElement<geometry_type>& element,
                const std::size_t block_size);

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

  /// String identifying the finite element
  /// @return Element signature
  /// @warning The function is provided for convenience, but it should
  /// not be relied upon for determining the element type. Use other
  /// functions, commonly returning enums, to determine element
  /// properties.
  std::string signature() const noexcept;

  /// Cell shape
  /// @return Element cell shape
  mesh::CellType cell_shape() const noexcept;

  /// Dimension of the finite element function space (the number of
  /// degrees-of-freedom for the element)
  /// @return Dimension of the finite element space
  int space_dimension() const noexcept;

  /// Block size of the finite element function space. For
  /// BlockedElements, this is the number of DOFs
  /// colocated at each DOF point. For other elements, this is always 1.
  /// @return Block size of the finite element space
  int block_size() const noexcept;

  /// The value size, e.g. 1 for a scalar function, 2 for a 2D vector, 9
  /// for a second-order tensor in 3D, for the reference element
  /// @return The value size for the reference element
  int reference_value_size() const;

  /// The reference value shape
  std::span<const std::size_t> reference_value_shape() const;

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

  /// Return underlying basix element (if it exists)
  const basix::FiniteElement<geometry_type>& basix_element() const;

  /// Get the map type used by the element
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

  /// @brief Return a function that applies DOF transformation operator
  /// `T to some data.
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
  /// @param[in] ttype The transformation type
  /// @param[in] scalar_element Indicates whether the scalar
  /// transformations should be returned for a vector element
  template <typename U>
  std::function<void(std::span<U>, std::span<const std::uint32_t>, std::int32_t,
                     int)>
  dof_transformation_function(doftransform ttype = doftransform::standard,
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

    if (_sub_elements.size() != 0)
    {
      if (_is_mixed)
      {
        // Mixed element
        std::vector<std::function<void(
            std::span<U>, std::span<const std::uint32_t>, std::int32_t, int)>>
            sub_element_functions;
        std::vector<int> dims;
        for (std::size_t i = 0; i < _sub_elements.size(); ++i)
        {
          sub_element_functions.push_back(
              _sub_elements[i]->template dof_transformation_function<U>(ttype));
          dims.push_back(_sub_elements[i]->space_dimension());
        }

        return [dims, sub_element_functions](
                   std::span<U> data, std::span<const std::uint32_t> cell_info,
                   std::int32_t cell, int block_size)
        {
          std::size_t offset = 0;
          for (std::size_t e = 0; e < sub_element_functions.size(); ++e)
          {
            const std::size_t width = dims[e] * block_size;
            sub_element_functions[e](data.subspan(offset, width), cell_info,
                                     cell, block_size);
            offset += width;
          }
        };
      }
      else if (!scalar_element)
      {
        // Vector element
        const std::function<void(std::span<U>, std::span<const std::uint32_t>,
                                 std::int32_t, int)>
            sub_function
            = _sub_elements[0]->template dof_transformation_function<U>(ttype);
        const int ebs = _bs;
        return [ebs, sub_function](std::span<U> data,
                                   std::span<const std::uint32_t> cell_info,
                                   std::int32_t cell, int data_block_size)
        { sub_function(data, cell_info, cell, ebs * data_block_size); };
      }
    }
    switch (ttype)
    {
    case doftransform::inverse_transpose:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      {
        pre_apply_inverse_transpose_dof_transformation(data, cell_info[cell],
                                                       block_size);
      };
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
  /// transposed data.
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
  /// @param[in] ttype The transformation type
  /// @param[in] scalar_element Indicated whether the scalar
  /// transformations should be returned for a vector element
  template <typename U>
  std::function<void(std::span<U>, std::span<const std::uint32_t>, std::int32_t,
                     int)>
  get_post_dof_transformation_function(doftransform ttype
                                       = doftransform::standard,
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
    else if (_sub_elements.size() != 0)
    {
      if (_is_mixed)
      {
        // Mixed element
        std::vector<std::function<void(
            std::span<U>, std::span<const std::uint32_t>, std::int32_t, int)>>
            sub_element_functions;
        for (std::size_t i = 0; i < _sub_elements.size(); ++i)
        {
          sub_element_functions.push_back(
              _sub_elements[i]
                  ->template get_post_dof_transformation_function<U>(ttype));
        }

        return [this, sub_element_functions](
                   std::span<U> data, std::span<const std::uint32_t> cell_info,
                   std::int32_t cell, int block_size)
        {
          std::size_t offset = 0;
          for (std::size_t e = 0; e < sub_element_functions.size(); ++e)
          {
            sub_element_functions[e](data.subspan(offset, data.size() - offset),
                                     cell_info, cell, block_size);
            offset += _sub_elements[e]->space_dimension();
          }
        };
      }
      else if (!scalar_element)
      {
        // Vector element
        const std::function<void(std::span<U>, std::span<const std::uint32_t>,
                                 std::int32_t, int)>
            sub_function
            = _sub_elements[0]->template dof_transformation_function<U>(ttype);
        return [this, sub_function](std::span<U> data,
                                    std::span<const std::uint32_t> cell_info,
                                    std::int32_t cell, int data_block_size)
        {
          const int ebs = block_size();
          const std::size_t dof_count = data.size() / data_block_size;
          for (int block = 0; block < data_block_size; ++block)
          {
            sub_function(data.subspan(block * dof_count, dof_count), cell_info,
                         cell, ebs);
          }
        };
      }
    }

    switch (ttype)
    {
    case doftransform::inverse_transpose:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      { Tt_inv_post_apply(data, cell_info[cell], block_size); };
    case doftransform::transpose:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      { Tt_post_apply(data, cell_info[cell], block_size); };
    case doftransform::inverse:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      { Tinv_post_apply(data, cell_info[cell], block_size); };
    case doftransform::standard:
      return [this](std::span<U> data, std::span<const std::uint32_t> cell_info,
                    std::int32_t cell, int block_size)
      { T_post_apply(data, cell_info[cell], block_size); };
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

  /// @brief Apply inverse transpose transformation to some data.
  ///
  /// For VectorElements, this applies the transformations for the
  /// scalar subelement.
  ///
  /// @param[in,out] data The data to be transformed. This data is
  /// flattened with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell.
  /// @param[in] n Block_size of the input data.
  template <typename U>
  void pre_apply_inverse_transpose_dof_transformation(
      std::span<U> data, std::uint32_t cell_permutation, int n) const
  {
    assert(_element);
    _element->Tt_inv_apply(data, n, cell_permutation);
  }

  /// @brief Apply the transpose of the operator applied by T_apply().
  ///
  /// The transformation
  /// \f[
  ///  v = T^{T} u
  /// \f]
  /// is performed in-place.
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
  /// The transformation
  /// \f[
  ///  v = T^{-1} u
  /// \f]
  /// is performed in-place.
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

  /// @brief Apply DOF transformation to some transposed data.
  ///
  /// @param[in,out] data The data to be transformed. This data is
  /// flattened with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] n Block size of the input data
  template <typename U>
  void T_post_apply(std::span<U> data, std::uint32_t cell_permutation,
                    int n) const
  {
    assert(_element);
    _element->T_post_apply(data, n, cell_permutation);
  }

  /// @brief Apply inverse of DOF transformation to some transposed
  /// data.
  ///
  /// @param[in,out] data Data to be transformed. This data is flattened
  /// with row-major layout, `shape=(num_dofs, block_size)`/
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] n Block size of the input data
  template <typename U>
  void Tinv_post_apply(std::span<U> data, std::uint32_t cell_permutation,
                       int n) const
  {
    assert(_element);
    _element->Tinv_post_apply(data, n, cell_permutation);
  }

  /// @brief Apply transpose of transformation to some transposed data.
  ///
  /// @param[in,out] data Data to be transformed. This data is flattened
  /// with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] n Block size of the input data.
  template <typename U>
  void Tt_post_apply(std::span<U> data, std::uint32_t cell_permutation,
                     int n) const
  {
    assert(_element);
    _element->Tt_post_apply(data, n, cell_permutation);
  }

  /// @brief Apply inverse transpose transformation to some transposed
  /// data.
  ///
  /// @param[in,out] data Data to be transformed. This data is flattened
  /// with row-major layout, `shape=(num_dofs, block_size)`.
  /// @param[in] cell_permutation Permutation data for the cell.
  /// @param[in] n Block size of the input data.
  template <typename U>
  void Tt_inv_post_apply(std::span<U> data, std::uint32_t cell_permutation,
                         int n) const
  {
    assert(_element);
    _element->Tt_inv_post_apply(data, n, cell_permutation);
  }

  /// @brief Permute the DOFs of the element.
  ///
  /// @param[in,out] doflist The numbers of the DOFs, a span of length
  /// `num_dofs`.
  /// @param[in] cell_permutation Permutation data for the cell.
  void permute(std::span<std::int32_t> doflist,
               std::uint32_t cell_permutation) const;

  /// @brief Unpermute the DOFs of the element.
  ///
  /// @param[in,out] doflist Numbers of the DOFs, a span of length
  /// `num_dofs`.
  /// @param[in] cell_permutation Permutation data for the cell.
  void permute_inv(std::span<std::int32_t> doflist,
                   std::uint32_t cell_permutation) const;

  /// @brief Return a function that applies DOF permutation to some
  /// data.
  ///
  /// The signature of the returned function has three arguments:
  /// - [in,out] doflist The numbers of the DOFs, a span of length num_dofs
  /// - [in] cell_permutation Permutation data for the cell
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] inverse Indicates whether the inverse transformations
  /// should be returned.
  /// @param[in] scalar_element Indicates whether the scalar
  /// transformations should be returned for a vector element.
  std::function<void(std::span<std::int32_t>, std::uint32_t)>
  get_dof_permutation_function(bool inverse = false,
                               bool scalar_element = false) const;

private:
  std::string _signature;

  mesh::CellType _cell_shape;

  int _space_dim;

  // List of sub-elements (if any)
  std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>
      _sub_elements;

  // Dimension of each value space
  std::vector<std::size_t> _reference_value_shape;

  // Block size for BlockedElements. This gives the
  // number of DOFs co-located at each dof 'point'.
  int _bs;

  // Indicate whether this is a mixed element
  bool _is_mixed;

  // Indicate whether the element needs permutations or transformations
  bool _needs_dof_permutations;
  bool _needs_dof_transformations;

  // Basix Element (nullptr for mixed elements)
  std::unique_ptr<basix::FiniteElement<geometry_type>> _element;

  // Quadrature points of a quadrature element (0 dimensional array for
  // all elements except quadrature elements)
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>> _points;
};
} // namespace dolfinx::fem
