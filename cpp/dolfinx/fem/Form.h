// Copyright (C) 2007-2014 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FormCoefficients.h"
#include "FormIntegrals.h"
#include <functional>
#include <map>
#include <memory>
#include <petscsys.h>
#include <set>
#include <string>
#include <vector>

// Forward declaration
struct ufc_form;

namespace dolfinx
{

namespace fem
{
class CoordinateElement;
}

namespace function
{
class Constant;
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

/// Base class for variational forms
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

class Form
{
public:
  /// Create form
  ///
  /// @param[in] function_spaces Function Spaces
  /// @param[in] integrals
  /// @param[in] coefficients
  /// @param[in] constants
  ///            Vector of pairs (name, constant). The index in the vector
  ///            is the position of the constant in the original
  ///            (nonsimplified) form.
  /// @param[in] coord_mapping Coordinate mapping
  Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>&
           function_spaces,
       const FormIntegrals& integrals, const FormCoefficients& coefficients,
       const std::vector<
           std::pair<std::string, std::shared_ptr<const function::Constant>>>
           constants,
       std::shared_ptr<const CoordinateElement> coord_mapping);

  /// Create form (no UFC integrals). Integrals can be attached later
  /// using FormIntegrals::set_cell_tabulate_tensor.
  /// @warning Experimental
  ///
  /// @param[in] function_spaces Vector of function spaces
  Form(const std::vector<std::shared_ptr<const function::FunctionSpace>>&
           function_spaces);

  /// Move constructor
  Form(Form&& form) = default;

  /// Destructor
  virtual ~Form() = default;

  /// Rank of form (bilinear form = 2, linear form = 1, functional = 0,
  /// etc)
  /// @return The rank of the form
  int rank() const;

  /// Set coefficient with given number (shared pointer version)
  /// @param[in] coefficients Map from coefficient index to the
  ///                         coefficient
  void set_coefficients(
      std::map<std::size_t, std::shared_ptr<const function::Function>>
          coefficients);

  /// Set coefficient with given name (shared pointer version)
  /// @param[in] coefficients Map from coefficient name to the
  ///                         coefficient
  void set_coefficients(
      std::map<std::string, std::shared_ptr<const function::Function>>
          coefficients);

  /// Return original coefficient position for each coefficient (0 <= i
  /// < n)
  /// @return The position of coefficient i in original ufl form
  ///         coefficients.
  int original_coefficient_position(int i) const;

  /// Set constants based on their names
  ///
  /// This method is used in command-line workflow, when users set
  /// constants to the form in cpp file.
  ///
  /// Names of the constants must agree with their names in UFL file.
  void
  set_constants(std::map<std::string, std::shared_ptr<const function::Constant>>
                    constants);

  /// Set constants based on their order (without names)
  ///
  /// This method is used in Python workflow, when constants are
  /// automatically attached to the form based on their order in the
  /// original form.
  ///
  /// The order of constants must match their order in original ufl
  /// Form.
  void set_constants(
      std::vector<std::shared_ptr<const function::Constant>> constants);

  /// Check if all constants associated with the form have been set
  /// @return True if all Form constants have been set
  bool all_constants_set() const;

  /// Return names of any constants that have not been set
  /// @return Names of unset constants
  std::set<std::string> get_unset_constants() const;

  /// Set mesh, necessary for functionals when there are no function
  /// spaces
  /// @param[in] mesh The mesh
  void set_mesh(std::shared_ptr<const mesh::Mesh> mesh);

  /// Extract common mesh from form
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const;

  /// Return function space for given argument
  /// @param[in] i Index of the argument
  /// @return Function space
  std::shared_ptr<const function::FunctionSpace> function_space(int i) const;

  /// Register the function for 'tabulate_tensor' for cell integral i
  void set_tabulate_tensor(
      FormIntegrals::Type type, int i,
      std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                         const double*, const int*, const std::uint8_t*,
                         const std::uint32_t)>
          fn);

  /// Set cell domains
  /// @param[in] cell_domains The cell domains
  void set_cell_domains(const mesh::MeshTags<int>& cell_domains);

  /// Set exterior facet domains
  /// @param[in] exterior_facet_domains The exterior facet domains
  void
  set_exterior_facet_domains(const mesh::MeshTags<int>& exterior_facet_domains);

  /// Set interior facet domains
  /// @param[in] interior_facet_domains The interior facet domains
  void
  set_interior_facet_domains(const mesh::MeshTags<int>& interior_facet_domains);

  /// Set vertex domains
  /// @param[in] vertex_domains The vertex domains.
  void set_vertex_domains(const mesh::MeshTags<int>& vertex_domains);

  /// Access coefficients
  FormCoefficients& coefficients();

  /// Access coefficients
  const FormCoefficients& coefficients() const;

  /// Access form integrals
  const FormIntegrals& integrals() const;

  /// Access constants
  /// @return Vector of attached constants with their names. Names are
  ///         used to set constants in user's c++ code. Index in the
  ///         vector is the position of the constant in the original
  ///         (nonsimplified) form.
  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant>>>&
  constants() const;

  /// Get coordinate_mapping
  /// @warning Experimental
  std::shared_ptr<const fem::CoordinateElement> coordinate_mapping() const;

private:
  // Integrals associated with the Form
  FormIntegrals _integrals;

  // Coefficients associated with the Form
  FormCoefficients _coefficients;

  // Constants associated with the Form
  std::vector<std::pair<std::string, std::shared_ptr<const function::Constant>>>
      _constants;

  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const function::FunctionSpace>> _function_spaces;

  // The mesh (needed for functionals when we don't have any spaces)
  std::shared_ptr<const mesh::Mesh> _mesh;

  // Coordinate_mapping
  std::shared_ptr<const fem::CoordinateElement> _coord_mapping;
};
} // namespace fem
} // namespace dolfinx
