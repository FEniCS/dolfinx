// Copyright (C) 2007-2014 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FormCoefficients.h"
#include "FormIntegrals.h"
// #include "utils.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <ufc.h>
#include <vector>

namespace dolfinx
{

namespace function
{
template <typename T>
class Constant;
template <typename T>
class Function;
class FunctionSpace;
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
  /// @param[in] constants Vector of pairs (name, constant). The index
  ///   in the vector is the position of the constant in the original
  ///   (nonsimplified) form.
  Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>&
           function_spaces,
       const FormIntegrals<T>& integrals,
       const FormCoefficients<T>& coefficients,
       const std::vector<
           std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>
           constants)
      : _integrals(integrals), _coefficients(coefficients),
        _constants(constants), _function_spaces(function_spaces)
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
  explicit Form(
      const std::vector<std::shared_ptr<const function::FunctionSpace>>&
          function_spaces)
      : Form(function_spaces, FormIntegrals<T>(), FormCoefficients<T>({}), {})
  {
    // Do nothing
  }

  /// Move constructor
  Form(Form&& form) = default;

  /// Destructor
  virtual ~Form() = default;

  /// Rank of the form (bilinear form = 2, linear form = 1, functional =
  /// 0, etc)
  /// @return The rank of the form
  int rank() const { return _function_spaces.size(); }

  /// Set coefficient with given name
  /// @param[in] coefficients Map from coefficient name to the
  ///   coefficient
  void set_coefficients(
      const std::map<std::string, std::shared_ptr<const function::Function<T>>>&
          coefficients)
  {
    for (const auto& c : coefficients)
      _coefficients.set(c.first, c.second);
  }

  /// Set constants based on their names. Names of the constants must
  /// agree with their names in UFL file.
  void set_constants(
      const std::map<std::string, std::shared_ptr<const function::Constant<T>>>&
          constants)
  {
    for (auto const& constant : constants)
    {
      // Find matching string in existing constants
      const std::string name = constant.first;
      const auto it = std::find_if(
          _constants.begin(), _constants.end(),
          [&](const std::pair<
              std::string, std::shared_ptr<const function::Constant<T>>>& q) {
            return (q.first == name);
          });
      if (it != _constants.end())
        it->second = constant.second;
      else
        throw std::runtime_error("Constant '" + name + "' not found in form");
    }
  }

  /// Check if all constants associated with the form have been set
  /// @return True if all Form constants have been set
  bool all_constants_set() const
  {
    for (const auto& constant : _constants)
      if (!constant.second)
        return false;
    return true;
  }

  /// Return names of any constants that have not been set
  /// @return Names of unset constants
  std::set<std::string> get_unset_constants() const
  {
    std::set<std::string> unset;
    for (const auto& constant : _constants)
      if (!constant.second)
        unset.insert(constant.first);
    return unset;
  }

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
  FormCoefficients<T>& coefficients() { return _coefficients; }

  /// Access coefficients
  const FormCoefficients<T>& coefficients() const { return _coefficients; }

  /// Access form integrals
  const FormIntegrals<T>& integrals() const { return _integrals; }

  /// Access constants
  /// @return Vector of attached constants with their names. Names are
  ///   used to set constants in user's c++ code. Index in the vector is
  ///   the position of the constant in the original (nonsimplified) form.
  std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>&
  constants()
  {
    return _constants;
  }

  /// Access constants
  /// @return Vector of attached constants with their names. Names are
  ///   used to set constants in user's c++ code. Index in the vector is
  ///   the position of the constant in the original (nonsimplified) form.
  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>&
  constants() const
  {
    return _constants;
  }

private:
  // Integrals associated with the Form
  FormIntegrals<T> _integrals;

  // Coefficients associated with the Form
  FormCoefficients<T> _coefficients;

  // Constants associated with the Form
  std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>
      _constants;

  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const function::FunctionSpace>> _function_spaces;

  // The mesh (needed for functionals when we don't have any spaces)
  std::shared_ptr<const mesh::Mesh> _mesh;
};

/// Extract coefficients from a UFC form
template <typename T>
std::vector<
    std::tuple<int, std::string, std::shared_ptr<function::Function<T>>>>
get_coeffs_from_ufc_form(const ufc_form& ufc_form)
{
  std::vector<
      std::tuple<int, std::string, std::shared_ptr<function::Function<T>>>>
      coeffs;
  const char** names = ufc_form.coefficient_name_map();
  for (int i = 0; i < ufc_form.num_coefficients; ++i)
  {
    coeffs.emplace_back(ufc_form.original_coefficient_position(i), names[i],
                        nullptr);
  }
  return coeffs;
}

/// Extract coefficients from a UFC form
template <typename T>
std::vector<
    std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>
get_constants_from_ufc_form(const ufc_form& ufc_form)
{
  std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>
      constants;
  const char** names = ufc_form.constant_name_map();
  for (int i = 0; i < ufc_form.num_constants; ++i)
    constants.emplace_back(names[i], nullptr);
  return constants;
}

/// Create a Form from UFC input
/// @param[in] ufc_form The UFC form
/// @param[in] spaces Vector of function spaces
template <typename T>
Form<T> create_form(
    const ufc_form& ufc_form,
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces)
{
  assert(ufc_form.rank == (int)spaces.size());

  // Check argument function spaces
  for (std::size_t i = 0; i < spaces.size(); ++i)
  {
    assert(spaces[i]->element());
    std::unique_ptr<ufc_finite_element, decltype(free)*> ufc_element(
        ufc_form.create_finite_element(i), free);
    assert(ufc_element);
    if (std::string(ufc_element->signature)
        != spaces[i]->element()->signature())
    {
      throw std::runtime_error(
          "Cannot create form. Wrong type of function space for argument.");
    }
  }

  // Get list of integral IDs, and load tabulate tensor into memory for each
  FormIntegrals<T> integrals;

  std::vector<int> cell_integral_ids(ufc_form.num_cell_integrals);
  ufc_form.get_cell_integral_ids(cell_integral_ids.data());
  for (int id : cell_integral_ids)
  {
    ufc_integral* cell_integral = ufc_form.create_cell_integral(id);
    assert(cell_integral);
    integrals.set_tabulate_tensor(IntegralType::cell, id,
                                  cell_integral->tabulate_tensor);
    std::free(cell_integral);
  }

  // FIXME: Can this be handled better?
  // FIXME: Handle forms with no space
  if (ufc_form.num_exterior_facet_integrals > 0
      or ufc_form.num_interior_facet_integrals > 0)
  {
    if (!spaces.empty())
    {
      auto mesh = spaces[0]->mesh();
      const int tdim = mesh->topology().dim();
      spaces[0]->mesh()->topology_mutable().create_entities(tdim - 1);
    }
  }

  std::vector<int> exterior_facet_integral_ids(
      ufc_form.num_exterior_facet_integrals);
  ufc_form.get_exterior_facet_integral_ids(exterior_facet_integral_ids.data());
  for (int id : exterior_facet_integral_ids)
  {
    ufc_integral* exterior_facet_integral
        = ufc_form.create_exterior_facet_integral(id);
    assert(exterior_facet_integral);
    integrals.set_tabulate_tensor(IntegralType::exterior_facet, id,
                                  exterior_facet_integral->tabulate_tensor);
    std::free(exterior_facet_integral);
  }

  std::vector<int> interior_facet_integral_ids(
      ufc_form.num_interior_facet_integrals);
  ufc_form.get_interior_facet_integral_ids(interior_facet_integral_ids.data());
  for (int id : interior_facet_integral_ids)
  {
    ufc_integral* interior_facet_integral
        = ufc_form.create_interior_facet_integral(id);
    assert(interior_facet_integral);
    integrals.set_tabulate_tensor(IntegralType::interior_facet, id,
                                  interior_facet_integral->tabulate_tensor);
    std::free(interior_facet_integral);
  }

  // Not currently working
  std::vector<int> vertex_integral_ids(ufc_form.num_vertex_integrals);
  ufc_form.get_vertex_integral_ids(vertex_integral_ids.data());
  if (vertex_integral_ids.size() > 0)
  {
    throw std::runtime_error(
        "Vertex integrals not supported. Under development.");
  }

  return fem::Form(
      spaces, integrals,
      FormCoefficients<T>(fem::get_coeffs_from_ufc_form<T>(ufc_form)),
      fem::get_constants_from_ufc_form<T>(ufc_form));
}

/// Create a form from a form_create function returning a pointer to a
/// ufc_form, taking care of memory allocation
/// @param[in] fptr pointer to a function returning a pointer to
///    ufc_form
/// @param[in] spaces function spaces
/// @return Form
template <typename T>
std::shared_ptr<Form<T>> create_form(
    ufc_form* (*fptr)(),
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces)
{
  ufc_form* form = fptr();
  auto L = std::make_shared<fem::Form<T>>(
      dolfinx::fem::create_form<T>(*form, spaces));
  std::free(form);
  return L;
}

} // namespace fem
} // namespace dolfinx
