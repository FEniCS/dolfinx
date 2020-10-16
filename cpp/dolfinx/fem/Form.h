// Copyright (C) 2007-2014 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FormIntegrals.h"
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/FunctionSpace.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dolfinx
{

namespace function
{
template <typename T>
class Constant;
template <typename T>
class Function;
} // namespace function

namespace mesh
{
class Mesh;
template <typename T>
class MeshTags;
} // namespace mesh

namespace fem
{

/// Class for variational forms
///
/// A note on the order of trial and test spaces: FEniCS numbers
/// argument spaces starting with the leading dimension of the
/// corresponding tensor (matrix). In other words, the test space is
/// numbered 0 and the trial space is numbered 1. However, in order to
/// have a notation that agrees with most existing finite element
/// literature, in particular
///
///  \f[   a = a(u, v)        \f]
///
/// the spaces are numbered from right to left
///
///  \f[   a: V_1 \times V_0 \rightarrow \mathbb{R}  \f]
///
/// This is reflected in the ordering of the spaces that should be
/// supplied to generated subclasses. In particular, when a bilinear
/// form is initialized, it should be initialized as `a(V_1, V_0) =
/// ...`, where `V_1` is the trial space and `V_0` is the test space.
/// However, when a form is initialized by a list of argument spaces
/// (the variable `function_spaces` in the constructors below), the list
/// of spaces should start with space number 0 (the test space) and then
/// space number 1 (the trial space).

template <typename T>
class Form
{
public:
  /// Create form
  ///
  /// @param[in] function_spaces Function Spaces
  /// @param[in] integrals
  /// @param[in] coefficients
  /// @param[in] constants Constants in the Form
  Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>&
           function_spaces,
       const FormIntegrals<T>& integrals,
       const std::vector<std::shared_ptr<const function::Function<T>>>&
           coefficients,
       const std::vector<std::shared_ptr<const function::Constant<T>>>&
           constants)
      : _function_spaces(function_spaces), _integrals(integrals),
        _coefficients(coefficients), _constants(constants)
  {
    // Set _mesh from function::FunctionSpace, and check they are the same
    if (!function_spaces.empty())
      _mesh = function_spaces[0]->mesh();
    for (const auto& V : function_spaces)
      if (_mesh != V->mesh())
        throw std::runtime_error("Incompatible mesh");

    // Set markers for default integrals
    if (_mesh)
      _integrals.set_default_domains(*_mesh);
  }

  /// @warning Experimental
  ///
  /// Create form (no UFC integrals). Integrals can be attached later
  /// using FormIntegrals::set_cell_tabulate_tensor.
  ///
  /// @param[in] function_spaces Vector of function spaces
  /// @param[in] coefficients
  /// @param[in] need_mesh_permutation_data Set to true if mesh entity
  ///   permutation data is required
  Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>&
           function_spaces,
       const std::vector<std::shared_ptr<const function::Function<T>>>&
           coefficients,
       bool need_mesh_permutation_data)
      : Form(function_spaces, FormIntegrals<T>({}, need_mesh_permutation_data),
             coefficients, {})
  {
    // Do nothing
  }

  /// Copy constructor
  Form(const Form& form) = delete;

  /// Move constructor
  Form(Form&& form) = default;

  /// Destructor
  virtual ~Form() = default;

  /// Rank of the form (bilinear form = 2, linear form = 1, functional =
  /// 0, etc)
  /// @return The rank of the form
  int rank() const { return _function_spaces.size(); }

  /// @todo Remove this function and make sure the mesh can be set via
  /// the constructor
  ///
  /// Set mesh, necessary for functionals when there are no function
  /// spaces
  /// @param[in] mesh The mesh
  void set_mesh(const std::shared_ptr<const mesh::Mesh>& mesh)
  {
    _mesh = mesh;
    // Set markers for default integrals
    _integrals.set_default_domains(*_mesh);
  }

  /// Extract common mesh from form
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const { return _mesh; }

  /// Return function space for given argument
  /// @param[in] i Index of the argument
  /// @return Function space
  std::shared_ptr<const function::FunctionSpace> function_space(int i) const
  {
    return _function_spaces.at(i);
  }

  /// Return function spaces for all arguments
  /// @return Function spaces
  std::vector<std::shared_ptr<const function::FunctionSpace>>
  function_spaces() const
  {
    return _function_spaces;
  }

  /// Register the function for 'tabulate_tensor' for cell integral i
  void set_tabulate_tensor(
      IntegralType type, int i,
      const std::function<void(T*, const T*, const T*, const double*,
                               const int*, const std::uint8_t*,
                               const std::uint32_t)>& fn)
  {
    _integrals.set_tabulate_tensor(type, i, fn);
    if (i == -1 and _mesh)
      _integrals.set_default_domains(*_mesh);
  }

  /// Access coefficients
  const std::vector<std::shared_ptr<const function::Function<T>>>
  coefficients() const
  {
    return _coefficients;
  }

  /// Offset for each coefficient expansion array on a cell. Used to
  /// pack data for multiple coefficients in a flat array. The last
  /// entry is the size required to store all coefficients.
  std::vector<int> coefficient_offsets() const
  {
    std::vector<int> n{0};
    for (const auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// Access form integrals
  const FormIntegrals<T>& integrals() const { return _integrals; }

  /// Access constants
  /// @return Vector of attached constants with their names. Names are
  ///   used to set constants in user's c++ code. Index in the vector is
  ///   the position of the constant in the original (nonsimplified) form.
  const std::vector<std::shared_ptr<const function::Constant<T>>>&
  constants() const
  {
    return _constants;
  }

private:
  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const function::FunctionSpace>> _function_spaces;

  // Integrals associated with the Form
  FormIntegrals<T> _integrals;

  // Form coefficients
  std::vector<std::shared_ptr<const function::Function<T>>> _coefficients;

  // Constants associated with the Form
  std::vector<std::shared_ptr<const function::Constant<T>>> _constants;

  // The mesh (needed for functionals when we don't have any spaces)
  std::shared_ptr<const mesh::Mesh> _mesh;
};

} // namespace fem
} // namespace dolfinx
