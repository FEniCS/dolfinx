// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/finite-element.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

struct ufc_finite_element;

namespace dolfinx::fem
{
/// Finite Element, containing the dof layout on a reference element,
/// and various methods for evaluating and transforming the basis.
class FiniteElement
{
public:
  /// Create finite element from UFC finite element
  /// @param[in] ufc_element UFC finite element
  explicit FiniteElement(const ufc_finite_element& ufc_element);

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

  /// String identifying the finite element
  /// @return Element signature
  /// @note The function is provided for convenience, but it should not
  /// be relied upon for determining the element type. Use other
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
  /// VectorElements and TensorElements, this is the number of DOFs
  /// colocated at each DOF point. For other elements, this is always 1.
  /// @return Block size of the finite element space
  int block_size() const noexcept;

  /// The value size, e.g. 1 for a scalar function, 2 for a 2D vector
  /// @return The value size
  int value_size() const noexcept;

  /// The value size, e.g. 1 for a scalar function, 2 for a 2D vector
  /// for the reference element
  /// @return The value size for the reference element
  int reference_value_size() const noexcept;

  /// Rank of the value space
  /// @return The value rank
  int value_rank() const noexcept;

  /// Return the dimension of the value space for axis i
  int value_dimension(int i) const;

  /// The finite element family
  /// @return The string of the finite element family
  std::string family() const noexcept;

  /// Evaluate all derivatives of the basis functions up to given order at given
  /// points in reference cell
  /// @param[in,out] values Four dimensional xtensor that will be filled with
  /// the tabulated values. Should be of shape {num_derivatives, num_points,
  /// num_dofs, reference_value_size}
  /// @param[in] X Two dimensional xtensor of shape [num_points, geometric
  /// dimension] containing the points at the reference element
  /// @param[in] order The number of derivatives (up to and including this
  /// order) to tabulate for.
  void tabulate(xt::xtensor<double, 4>& values, const xt::xtensor<double, 2>& X,
                int order) const;

  /// Push basis functions forward to physical element
  /// @param[out] values Basis function values on the physical domain (ndim=3)
  /// @param[in] reference_values Basis function values on the reference
  /// cell (ndim=3)
  /// @param[in] J The Jacobian of the map (shape=(num_points, gdim, tdim))
  /// @param[in] detJ The determinant of the Jacobian
  /// @param[in] K The inverse of the Jacobian (shape=(num_points, tdim, gdim))
  template <typename U, typename V, typename W, typename X>
  constexpr void
  transform_reference_basis(U&& values, const V& reference_values, const W& J,
                            const xtl::span<const double>& detJ,
                            const X& K) const
  {
    assert(_element);
    _element->map_push_forward_m(reference_values, J, detJ, K, values);
  }

  /// Get the number of sub elements (for a mixed or blocked element)
  /// @return The number of sub elements
  int num_sub_elements() const noexcept;

  /// Check if element is a mixed element, i.e. composed of two or more
  /// elements of different types. A block element, e.g. a Lagrange
  /// element with block size > 1 is not considered mixed.
  /// @return True is element is mixed.
  bool is_mixed() const noexcept;

  /// Subelements (if any)
  const std::vector<std::shared_ptr<const FiniteElement>>&
  sub_elements() const noexcept;

  /// Return simple hash of the signature string
  std::size_t hash() const noexcept;

  /// Extract sub finite element for component
  std::shared_ptr<const FiniteElement>
  extract_sub_element(const std::vector<int>& component) const;

  /// Check if interpolation into the finite element space is an
  /// identity operation given the evaluation on an expression at
  /// specific points, i.e. the degree-of-freedom are equal to point
  /// evaluations. The function will return `true` for Lagrange
  /// elements.
  ///  @return True if interpolation is an identity operation
  bool interpolation_ident() const noexcept;

  /// Points on the reference cell at which an expression need to be
  /// evaluated in order to interpolate the expression in the finite
  /// element space. For Lagrange elements the points will just be the
  /// nodal positions. For other elements the points will typically be
  /// the quadrature points used to evaluate moment degrees of freedom.
  /// @return Points on the reference cell. Shape is (num_points, tdim).
  const xt::xtensor<double, 2>& interpolation_points() const;

  /// @todo Document shape/layout of @p values
  /// @todo Make the interpolating dofs in/out argument for efficiency
  /// as this function is often called from within tight loops
  /// @todo Consider handling block size > 1
  /// @todo Re-work for fields that require a pull-back, e.g. Piola
  /// mapped elements
  ///
  /// Interpolate a function in the finite element space on a cell.
  /// Given the evaluation of the function to be interpolated at points
  /// provided by @p FiniteElement::interpolation_points, it evaluates
  /// the degrees of freedom for the interpolant.
  ///
  /// @param[in] values The values of the function. It has shape
  /// (value_size, num_points), where `num_points` is the number of
  /// points given by FiniteElement::interpolation_points.
  /// @param[out] dofs The element degrees of freedom (interpolants) of
  /// the expression. The call must allocate the space. Is has
  template <typename T>
  constexpr void interpolate(const xt::xtensor<T, 2>& values,
                             const xtl::span<T>& dofs) const
  {
    if (!_element)
    {
      throw std::runtime_error("No underlying element for interpolation. "
                               "Cannot interpolate mixed elements directly.");
    }

    _element->interpolate(dofs, tcb::make_span(values), _bs);
  }

  /// Create a matrix that maps degrees of freedom from one element to
  /// this element (interpolation)
  /// @param[in] from The element to interpolate from
  /// @return Matrix operator that maps the 'from' degrees-of-freedom to
  /// the degrees-of-freedom of this element.
  /// @note Does not support mixed elements
  xt::xtensor<double, 2>
  create_interpolation_operator(const FiniteElement& from) const;

  /// Check if DOF transformations are needed for this element.
  ///
  /// DOF transformations will be needed for elements which might not be
  /// continuous when two neighbouring cells disagree on the orientation of
  /// a shared subentity, and when this cannot be corrected for by permuting
  /// the DOF numbering in the dofmap.
  ///
  /// For example, Raviart-Thomas elements will need DOF transformations,
  /// as the neighbouring cells may disagree on the orientation of a basis
  /// function, and this orientation cannot be corrected for by permuting
  /// the DOF numbers on each cell.
  ///
  /// @return True if DOF transformations are required
  bool needs_dof_transformations() const noexcept;

  /// Check if DOF permutations are needed for this element.
  ///
  /// DOF permutations will be needed for elements which might not be
  /// continuous when two neighbouring cells disagree on the orientation of
  /// a shared subentity, and when this can be corrected for by permuting the
  /// DOF numbering in the dofmap.
  ///
  /// For example, higher order Lagrange elements will need DOF permutations,
  /// as the arrangement of DOFs on a shared subentity may be different from the
  /// point of view of neighbouring cells, and this can be corrected for by
  /// permuting the DOF numbers on each cell.
  ///
  /// @return True if DOF transformations are required
  bool needs_dof_permutations() const noexcept;

  /// Return a function that applies DOF transformation to some data.
  ///
  /// The returned function will take three inputs:
  /// - [in,out] data The data to be transformed
  /// - [in] cell_info Permutation data for the cell
  /// - [in] cell The cell number
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] inverse Indicates whether the inverse transformations should be
  /// returned
  /// @param[in] transpose Indicates whether the transpose transformations
  /// should be returned
  /// @param[in] scalar_element Indicates whether the scalar transformations
  /// should be returned for a vector element
  template <typename T>
  std::function<void(const xtl::span<T>&, const xtl::span<const std::uint32_t>&,
                     std::int32_t, int)>
  get_dof_transformation_function(bool inverse = false, bool transpose = false,
                                  bool scalar_element = false) const
  {
    if (!needs_dof_transformations())
    {
      // If no permutation needed, return function that does nothing
      return [](const xtl::span<T>&, const xtl::span<const std::uint32_t>&,
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
        std::vector<std::function<void(const xtl::span<T>&,
                                       const xtl::span<const std::uint32_t>&,
                                       std::int32_t, int)>>
            sub_element_functions;
        std::vector<int> dims;
        for (std::size_t i = 0; i < _sub_elements.size(); ++i)
        {
          sub_element_functions.push_back(
              _sub_elements[i]->get_dof_transformation_function<T>(inverse,
                                                                   transpose));
          dims.push_back(_sub_elements[i]->space_dimension());
        }

        return [dims, sub_element_functions](
                   const xtl::span<T>& data,
                   const xtl::span<const std::uint32_t>& cell_info,
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
        const std::function<void(const xtl::span<T>&,
                                 const xtl::span<const std::uint32_t>&,
                                 std::int32_t, int)>
            sub_function = _sub_elements[0]->get_dof_transformation_function<T>(
                inverse, transpose);
        const int ebs = _bs;
        return
            [ebs, sub_function](const xtl::span<T>& data,
                                const xtl::span<const std::uint32_t>& cell_info,
                                std::int32_t cell, int data_block_size)
        { sub_function(data, cell_info, cell, ebs * data_block_size); };
      }
    }
    if (transpose)
    {
      if (inverse)
      {
        return [this](const xtl::span<T>& data,
                      const xtl::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        {
          apply_inverse_transpose_dof_transformation(data, cell_info[cell],
                                                     block_size);
        };
      }
      else
      {
        return [this](const xtl::span<T>& data,
                      const xtl::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size) {
          apply_transpose_dof_transformation(data, cell_info[cell], block_size);
        };
      }
    }
    else
    {
      if (inverse)
      {
        return [this](const xtl::span<T>& data,
                      const xtl::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size) {
          apply_inverse_dof_transformation(data, cell_info[cell], block_size);
        };
      }
      else
      {
        return [this](const xtl::span<T>& data,
                      const xtl::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        { apply_dof_transformation(data, cell_info[cell], block_size); };
      }
    }
  }

  /// Return a function that applies DOF transformation to some
  /// transposed data
  ///
  /// The returned function will take three inputs:
  /// - [in,out] data The data to be transformed
  /// - [in] cell_info Permutation data for the cell
  /// - [in] cell The cell number
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] inverse Indicates whether the inverse transformations
  /// should be returned
  /// @param[in] transpose Indicates whether the transpose
  /// transformations should be returned
  /// @param[in] scalar_element Indicated whether the scalar
  /// transformations should be returned for a vector element
  template <typename T>
  std::function<void(const xtl::span<T>&, const xtl::span<const std::uint32_t>&,
                     std::int32_t, int)>
  get_dof_transformation_to_transpose_function(bool inverse = false,
                                               bool transpose = false,
                                               bool scalar_element
                                               = false) const
  {
    if (!needs_dof_transformations())
    {
      // If no permutation needed, return function that does nothing
      return [](const xtl::span<T>&, const xtl::span<const std::uint32_t>&,
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
        std::vector<std::function<void(const xtl::span<T>&,
                                       const xtl::span<const std::uint32_t>&,
                                       std::int32_t, int)>>
            sub_element_functions;
        for (std::size_t i = 0; i < _sub_elements.size(); ++i)
        {
          sub_element_functions.push_back(
              _sub_elements[i]->get_dof_transformation_to_transpose_function<T>(
                  inverse, transpose));
        }

        return [this, sub_element_functions](
                   const xtl::span<T>& data,
                   const xtl::span<const std::uint32_t>& cell_info,
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
        const std::function<void(const xtl::span<T>&,
                                 const xtl::span<const std::uint32_t>&,
                                 std::int32_t, int)>
            sub_function = _sub_elements[0]->get_dof_transformation_function<T>(
                inverse, transpose);
        return [this,
                sub_function](const xtl::span<T>& data,
                              const xtl::span<const std::uint32_t>& cell_info,
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
        return [this](const xtl::span<T>& data,
                      const xtl::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        {
          apply_inverse_transpose_dof_transformation_to_transpose(
              data, cell_info[cell], block_size);
        };
      }
      else
      {
        return [this](const xtl::span<T>& data,
                      const xtl::span<const std::uint32_t>& cell_info,
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
        return [this](const xtl::span<T>& data,
                      const xtl::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size)
        {
          apply_inverse_dof_transformation_to_transpose(data, cell_info[cell],
                                                        block_size);
        };
      }
      else
      {
        return [this](const xtl::span<T>& data,
                      const xtl::span<const std::uint32_t>& cell_info,
                      std::int32_t cell, int block_size) {
          apply_dof_transformation_to_transpose(data, cell_info[cell],
                                                block_size);
        };
      }
    }
  }

  /// Apply DOF transformation to some data
  ///
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_dof_transformation(const xtl::span<T>& data,
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
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void
  apply_inverse_transpose_dof_transformation(const xtl::span<T>& data,
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
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_transpose_dof_transformation(const xtl::span<T>& data,
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
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_inverse_dof_transformation(const xtl::span<T>& data,
                                        std::uint32_t cell_permutation,
                                        int block_size) const
  {
    assert(_element);
    _element->apply_inverse_dof_transformation(data, block_size,
                                               cell_permutation);
  }

  /// Apply DOF transformation to some tranposed data
  ///
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_dof_transformation_to_transpose(const xtl::span<T>& data,
                                             std::uint32_t cell_permutation,
                                             int block_size) const
  {
    assert(_element);
    _element->apply_dof_transformation_to_transpose(data, block_size,
                                                    cell_permutation);
  }

  /// Apply inverse of DOF transformation to some transposed data.
  ///
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void
  apply_inverse_dof_transformation_to_transpose(const xtl::span<T>& data,
                                                std::uint32_t cell_permutation,
                                                int block_size) const
  {
    assert(_element);
    _element->apply_inverse_dof_transformation_to_transpose(data, block_size,
                                                            cell_permutation);
  }

  /// Apply transpose of transformation to some transposed data.
  ///
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_transpose_dof_transformation_to_transpose(
      const xtl::span<T>& data, std::uint32_t cell_permutation,
      int block_size) const
  {
    assert(_element);
    _element->apply_transpose_dof_transformation_to_transpose(data, block_size,
                                                              cell_permutation);
  }

  /// Apply inverse transpose transformation to some transposed data
  ///
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_inverse_transpose_dof_transformation_to_transpose(
      const xtl::span<T>& data, std::uint32_t cell_permutation,
      int block_size) const
  {
    assert(_element);
    _element->apply_inverse_transpose_dof_transformation_to_transpose(
        data, block_size, cell_permutation);
  }

  /// Pull back data from the physical element to the reference element.
  /// It can process batches of points that share the same geometric
  /// map. @note This passes the inputs directly into the Basix
  /// `map_pull_back` function
  ///
  /// @param[in] u Data defined on the physical element. It must have
  /// dimension 3. The first index is for the geometric/map data, the
  /// second is the point index for points that share map data, and the
  /// third index is (vector) component, e.g. `u[i,:,:]` are points that
  /// are mapped by `J[i,:,:]`.
  /// @param[in] J The Jacobians. It must have dimension 3. The first
  /// index is for the ith Jacobian, i.e. J[i,:,:] is the ith Jacobian.
  /// @param[in] detJ The determinant of J. `detJ[i]` is
  /// `det(J[i,:,:])`. It must have dimension 1.
  /// @param[in] K The inverse of J, `K[i,:,:] = J[i,:,:]^-1`. It must
  /// have dimension 3.
  /// @param[out] U The input `u` mapped to the reference element. It
  /// must have dimension 3.
  template <typename O, typename P, typename Q, typename T, typename S>
  void map_pull_back(const O& u, const P& J, const Q& detJ, const T& K,
                     S&& U) const
  {
    assert(_element);
    _element->map_pull_back_m(u, J, detJ, K, U);
  }

  /// Permute the DOFs of the element
  ///
  /// @param[in,out] doflist The numbers of the DOFs
  /// @param[in] cell_permutation Permutation data for the cell
  void permute_dofs(const xtl::span<std::int32_t>& doflist,
                    std::uint32_t cell_permutation) const;

  /// Unpermute the DOFs of the element
  ///
  /// @param[in,out] doflist The numbers of the DOFs
  /// @param[in] cell_permutation Permutation data for the cell
  void unpermute_dofs(const xtl::span<std::int32_t>& doflist,
                      std::uint32_t cell_permutation) const;

  /// Return a function that applies DOF transformation to some data
  ///
  /// The returned function will take three inputs:
  /// - [in,out] data The data to be transformed
  /// - [in] cell_permutation Permutation data for the cell
  /// - [in] block_size The block_size of the input data
  ///
  /// @param[in] inverse Indicates whether the inverse transformations
  /// should be returned
  /// @param[in] scalar_element Indicated whether the scalar
  /// transformations should be returned for a vector element
  std::function<void(const xtl::span<std::int32_t>&, std::uint32_t)>
  get_dof_permutation_function(bool inverse = false,
                               bool scalar_element = false) const;

private:
  std::string _signature, _family;

  mesh::CellType _cell_shape;

  int _tdim, _space_dim, _value_size, _reference_value_size;

  // List of sub-elements (if any)
  std::vector<std::shared_ptr<const FiniteElement>> _sub_elements;

  // Simple hash of the signature string
  std::size_t _hash;

  // Dimension of each value space
  std::vector<int> _value_dimension;

  // Block size for VectorElements and TensorElements. This gives the
  // number of DOFs colocated at each point.
  int _bs;

  // Indicate whether the element needs permutations or transformations
  bool _needs_dof_permutations;
  bool _needs_dof_transformations;

  // Basix Element (nullptr for mixed elements)
  std::unique_ptr<basix::FiniteElement> _element;
};
} // namespace dolfinx::fem
