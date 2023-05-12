// Copyright (C) 2020-2021 Garth N. Wells and Matthew W. Scroggs
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <basix/finite-element.h>
#include <concepts>
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
  /// Create finite element from UFC finite element
  /// @param[in] e UFC finite element
  explicit FiniteElement(const ufcx_finite_element& e);

  /// Create finite element from a Basix finite element
  /// @param[in] element Basix finite element
  /// @param[in] bs The block size
  FiniteElement(const basix::FiniteElement<T>& element, int bs);

  /// Copy constructor
  FiniteElement(const FiniteElement& element) = delete;

  /// Move constructor
  FiniteElement(FiniteElement&& element) = default;

  /// Destructor
  virtual ~FiniteElement() = default;

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
  /// for a second-order tensor in 3D.
  /// @note The return value of this function is equivalent to
  /// `std::accumulate(value_shape().begin(), value_shape().end(), 1,
  /// std::multiplies{})`.
  /// @return The value size
  int value_size() const;

  /// The value size, e.g. 1 for a scalar function, 2 for a 2D vector, 9
  /// for a second-order tensor in 3D, for the reference element
  /// @return The value size for the reference element
  int reference_value_size() const;

  /// Shape of the value space. The rank is the size of the
  /// `value_shape`.
  std::span<const std::size_t> value_shape() const noexcept;

  /// The finite element family
  /// @return The string of the finite element family
  std::string family() const noexcept;

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
  void tabulate(std::span<T> values, std::span<const T> X,
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
  std::pair<std::vector<T>, std::array<std::size_t, 4>>
  tabulate(std::span<const T> X, std::array<std::size_t, 2> shape,
           int order) const;

  /// @brief Number of sub elements (for a mixed or blocked element)
  /// @return The number of sub elements
  int num_sub_elements() const noexcept;

  /// Check if element is a mixed element, i.e. composed of two or more
  /// elements of different types. A block element, e.g. a Lagrange
  /// element with block size > 1 is not considered mixed.
  /// @return True is element is mixed.
  bool is_mixed() const noexcept;

  /// Subelements (if any)
  const std::vector<std::shared_ptr<const FiniteElement<T>>>&
  sub_elements() const noexcept;

  /// Extract sub finite element for component
  std::shared_ptr<const FiniteElement<T>>
  extract_sub_element(const std::vector<int>& component) const;

  /// Return underlying basix element (if it exists)
  const basix::FiniteElement<T>& basix_element() const;

  /// Get the map type used by the element
  basix::maps::type map_type() const;

  /// Check if interpolation into the finite element space is an
  /// identity operation given the evaluation on an expression at
  /// specific points, i.e. the degree-of-freedom are equal to point
  /// evaluations. The function will return `true` for Lagrange
  /// elements.
  /// @return True if interpolation is an identity operation
  bool interpolation_ident() const noexcept;

  /// Check if the push forward/pull back map from the values on reference to
  /// the values on a physical cell for this element is the identity map.
  /// @return True if the map is the identity
  bool map_ident() const noexcept;

  /// @brief Points on the reference cell at which an expression needs to
  /// be evaluated in order to interpolate the expression in the finite
  /// element space.
  ///
  /// For Lagrange elements the points will just be the
  /// nodal positions. For other elements the points will typically be
  /// the quadrature points used to evaluate moment degrees of freedom.
  /// @return Interpolation point coordinates on the reference cell,
  /// returning the (0) coordinates data (row-major) storage and (1) the
  /// shape `(num_points, tdim)`.
  std::pair<std::vector<T>, std::array<std::size_t, 2>>
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
  std::pair<std::vector<T>, std::array<std::size_t, 2>>
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
  std::pair<std::vector<T>, std::array<std::size_t, 2>>
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

  /// Return a function that applies DOF transformation to some data.
  ///
  /// The returned function will take four inputs:
  /// - [in,out] data The data to be transformed. This data is flattened
  ///   with row-major layout, shape=(num_dofs, block_size)
  /// - [in] cell_info Permutation data for the cell. The size of this
  ///   is num_cells. For elements where no transformations are required,
  ///   an empty span can be passed in.
  /// - [in] cell The cell number
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] inverse Indicates whether the inverse transformations
  /// should be returned
  /// @param[in] transpose Indicates whether the transpose
  /// transformations should be returned
  /// @param[in] scalar_element Indicates whether the scalar
  /// transformations should be returned for a vector element
  template <typename U>
  std::function<void(const std::span<U>&, const std::span<const std::uint32_t>&,
                     std::int32_t, int)>
  get_dof_transformation_function(bool inverse = false, bool transpose = false,
                                  bool scalar_element = false) const
  {
    if (!needs_dof_transformations())
    {
      // If no permutation needed, return function that does nothing
      return [](const std::span<U>&, const std::span<const std::uint32_t>&,
                std::int32_t, int)
      {
        // Do nothing
      };
    }

    if (_sub_elements.size() != 0)
    {
      if (_bs == 1)
      {
        // Mixed element
        std::vector<std::function<void(const std::span<U>&,
                                       const std::span<const std::uint32_t>&,
                                       std::int32_t, int)>>
            sub_element_functions;
        std::vector<int> dims;
        for (std::size_t i = 0; i < _sub_elements.size(); ++i)
        {
          sub_element_functions.push_back(
              _sub_elements[i]->template get_dof_transformation_function<U>(
                  inverse, transpose));
          dims.push_back(_sub_elements[i]->space_dimension());
        }

        return [dims, sub_element_functions](
                   const std::span<U>& data,
                   const std::span<const std::uint32_t>& cell_info,
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
        const std::function<void(const std::span<U>&,
                                 const std::span<const std::uint32_t>&,
                                 std::int32_t, int)>
            sub_function
            = _sub_elements[0]->template get_dof_transformation_function<U>(
                inverse, transpose);
        const int ebs = _bs;
        return
            [ebs, sub_function](const std::span<U>& data,
                                const std::span<const std::uint32_t>& cell_info,
                                std::int32_t cell, int data_block_size)
        { sub_function(data, cell_info, cell, ebs * data_block_size); };
      }
    }
    if (transpose)
    {
      if (inverse)
      {
        return [this](const std::span<U>& data,
                      const std::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        {
          apply_inverse_transpose_dof_transformation(data, cell_info[cell],
                                                     block_size);
        };
      }
      else
      {
        return [this](const std::span<U>& data,
                      const std::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size) {
          apply_transpose_dof_transformation(data, cell_info[cell], block_size);
        };
      }
    }
    else
    {
      if (inverse)
      {
        return [this](const std::span<U>& data,
                      const std::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size) {
          apply_inverse_dof_transformation(data, cell_info[cell], block_size);
        };
      }
      else
      {
        return [this](const std::span<U>& data,
                      const std::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        { apply_dof_transformation(data, cell_info[cell], block_size); };
      }
    }
  }

  /// Return a function that applies DOF transformation to some
  /// transposed data
  ///
  /// The returned function will take three inputs:
  /// - [in,out] data The data to be transformed. This data is flattened
  ///   with row-major layout, shape=(num_dofs, block_size)
  /// - [in] cell_info Permutation data for the cell. The size of this
  ///   is num_cells. For elements where no transformations are required,
  ///   an empty span can be passed in.
  /// - [in] cell The cell number
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] inverse Indicates whether the inverse transformations
  /// should be returned
  /// @param[in] transpose Indicates whether the transpose
  /// transformations should be returned
  /// @param[in] scalar_element Indicated whether the scalar
  /// transformations should be returned for a vector element
  template <typename U>
  std::function<void(const std::span<U>&, const std::span<const std::uint32_t>&,
                     std::int32_t, int)>
  get_dof_transformation_to_transpose_function(bool inverse = false,
                                               bool transpose = false,
                                               bool scalar_element
                                               = false) const
  {
    if (!needs_dof_transformations())
    {
      // If no permutation needed, return function that does nothing
      return [](const std::span<U>&, const std::span<const std::uint32_t>&,
                std::int32_t, int)
      {
        // Do nothing
      };
    }
    else if (_sub_elements.size() != 0)
    {
      if (_bs == 1)
      {
        // Mixed element
        std::vector<std::function<void(const std::span<U>&,
                                       const std::span<const std::uint32_t>&,
                                       std::int32_t, int)>>
            sub_element_functions;
        for (std::size_t i = 0; i < _sub_elements.size(); ++i)
        {
          sub_element_functions.push_back(
              _sub_elements[i]
                  ->template get_dof_transformation_to_transpose_function<U>(
                      inverse, transpose));
        }

        return [this, sub_element_functions](
                   const std::span<U>& data,
                   const std::span<const std::uint32_t>& cell_info,
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
        const std::function<void(const std::span<U>&,
                                 const std::span<const std::uint32_t>&,
                                 std::int32_t, int)>
            sub_function
            = _sub_elements[0]->template get_dof_transformation_function<U>(
                inverse, transpose);
        return [this,
                sub_function](const std::span<U>& data,
                              const std::span<const std::uint32_t>& cell_info,
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

    if (transpose)
    {
      if (inverse)
      {
        return [this](const std::span<U>& data,
                      const std::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        {
          apply_inverse_transpose_dof_transformation_to_transpose(
              data, cell_info[cell], block_size);
        };
      }
      else
      {
        return [this](const std::span<U>& data,
                      const std::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        {
          apply_transpose_dof_transformation_to_transpose(data, cell_info[cell],
                                                          block_size);
        };
      }
    }
    else
    {
      if (inverse)
      {
        return [this](const std::span<U>& data,
                      const std::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        {
          apply_inverse_dof_transformation_to_transpose(data, cell_info[cell],
                                                        block_size);
        };
      }
      else
      {
        return [this](const std::span<U>& data,
                      const std::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size) {
          apply_dof_transformation_to_transpose(data, cell_info[cell],
                                                block_size);
        };
      }
    }
  }

  /// Apply DOF transformation to some data
  ///
  /// @param[in,out] data The data to be transformed. This data is flattened
  /// with row-major layout, shape=(num_dofs, block_size)
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename U>
  void apply_dof_transformation(const std::span<U>& data,
                                std::uint32_t cell_permutation,
                                int block_size) const
  {
    assert(_element);
    _element->apply_dof_transformation(data, block_size, cell_permutation);
  }

  /// Apply inverse transpose transformation to some data. For
  /// VectorElements, this applies the transformations for the scalar
  /// subelement.
  ///
  /// @param[in,out] data The data to be transformed. This data is flattened
  /// with row-major layout, shape=(num_dofs, block_size)
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename U>
  void
  apply_inverse_transpose_dof_transformation(const std::span<U>& data,
                                             std::uint32_t cell_permutation,
                                             int block_size) const
  {
    assert(_element);
    _element->apply_inverse_transpose_dof_transformation(data, block_size,
                                                         cell_permutation);
  }

  /// Apply transpose transformation to some data. For VectorElements,
  /// this applies the transformations for the scalar subelement.
  ///
  /// @param[in,out] data The data to be transformed. This data is flattened
  /// with row-major layout, shape=(num_dofs, block_size)
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename U>
  void apply_transpose_dof_transformation(const std::span<U>& data,
                                          std::uint32_t cell_permutation,
                                          int block_size) const
  {
    assert(_element);
    _element->apply_transpose_dof_transformation(data, block_size,
                                                 cell_permutation);
  }

  /// Apply inverse transformation to some data. For VectorElements,
  /// this applies the transformations for the scalar subelement.
  ///
  /// @param[in,out] data The data to be transformed. This data is flattened
  /// with row-major layout, shape=(num_dofs, block_size)
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename U>
  void apply_inverse_dof_transformation(const std::span<U>& data,
                                        std::uint32_t cell_permutation,
                                        int block_size) const
  {
    assert(_element);
    _element->apply_inverse_dof_transformation(data, block_size,
                                               cell_permutation);
  }

  /// Apply DOF transformation to some transposed data
  ///
  /// @param[in,out] data The data to be transformed. This data is flattened
  /// with row-major layout, shape=(num_dofs, block_size)
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename U>
  void apply_dof_transformation_to_transpose(const std::span<U>& data,
                                             std::uint32_t cell_permutation,
                                             int block_size) const
  {
    assert(_element);
    _element->apply_dof_transformation_to_transpose(data, block_size,
                                                    cell_permutation);
  }

  /// Apply inverse of DOF transformation to some transposed data.
  ///
  /// @param[in,out] data The data to be transformed. This data is flattened
  /// with row-major layout, shape=(num_dofs, block_size)
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename U>
  void
  apply_inverse_dof_transformation_to_transpose(const std::span<U>& data,
                                                std::uint32_t cell_permutation,
                                                int block_size) const
  {
    assert(_element);
    _element->apply_inverse_dof_transformation_to_transpose(data, block_size,
                                                            cell_permutation);
  }

  /// Apply transpose of transformation to some transposed data.
  ///
  /// @param[in,out] data The data to be transformed. This data is flattened
  /// with row-major layout, shape=(num_dofs, block_size)
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename U>
  void apply_transpose_dof_transformation_to_transpose(
      const std::span<U>& data, std::uint32_t cell_permutation,
      int block_size) const
  {
    assert(_element);
    _element->apply_transpose_dof_transformation_to_transpose(data, block_size,
                                                              cell_permutation);
  }

  /// Apply inverse transpose transformation to some transposed data
  ///
  /// @param[in,out] data The data to be transformed. This data is flattened
  /// with row-major layout, shape=(num_dofs, block_size)
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename U>
  void apply_inverse_transpose_dof_transformation_to_transpose(
      const std::span<U>& data, std::uint32_t cell_permutation,
      int block_size) const
  {
    assert(_element);
    _element->apply_inverse_transpose_dof_transformation_to_transpose(
        data, block_size, cell_permutation);
  }

  /// Permute the DOFs of the element
  ///
  /// @param[in,out] doflist The numbers of the DOFs, a span of length num_dofs
  /// @param[in] cell_permutation Permutation data for the cell
  void permute_dofs(const std::span<std::int32_t>& doflist,
                    std::uint32_t cell_permutation) const;

  /// Unpermute the DOFs of the element
  ///
  /// @param[in,out] doflist The numbers of the DOFs, a span of length num_dofs
  /// @param[in] cell_permutation Permutation data for the cell
  void unpermute_dofs(const std::span<std::int32_t>& doflist,
                      std::uint32_t cell_permutation) const;

  /// Return a function that applies DOF permutation to some data
  ///
  /// The returned function will take three inputs:
  /// - [in,out] doflist The numbers of the DOFs, a span of length num_dofs
  /// - [in] cell_permutation Permutation data for the cell
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] inverse Indicates whether the inverse transformations
  /// should be returned
  /// @param[in] scalar_element Indicated whether the scalar
  /// transformations should be returned for a vector element
  std::function<void(const std::span<std::int32_t>&, std::uint32_t)>
  get_dof_permutation_function(bool inverse = false,
                               bool scalar_element = false) const;

private:
  std::string _signature, _family;

  mesh::CellType _cell_shape;

  int _space_dim;

  // List of sub-elements (if any)
  std::vector<std::shared_ptr<const FiniteElement<T>>> _sub_elements;

  // Dimension of each value space
  std::vector<std::size_t> _value_shape;

  // Block size for BlockedElements. This gives the
  // number of DOFs co-located at each dof 'point'.
  int _bs;

  // Indicate whether the element needs permutations or transformations
  bool _needs_dof_permutations;
  bool _needs_dof_transformations;

  // Basix Element (nullptr for mixed elements)
  std::unique_ptr<basix::FiniteElement<T>> _element;
};
} // namespace dolfinx::fem
