// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfin
{
namespace mesh
{
template <typename T>
class MeshFunction;
class Mesh;
} // namespace mesh

namespace fem
{

// FIXME: This class would be greatly simplified if all integrals types
// (cell, facet, etc) were the same type.

/// Integrals of a Form, including those defined over cells, interior
/// and exterior facets, and vertices.
class FormIntegrals
{
public:
  /// Type of integral
  enum class Type : std::int8_t
  {
    cell = 0,
    exterior_facet = 1,
    interior_facet = 2,
    vertex = 3
  };

  /// Initialise the FormIntegrals as empty
  FormIntegrals();

  /// Get the function for 'tabulate_tensor' for cell integral i
  /// @param i
  ///    Integral number
  /// @returns std::function
  ///    Function to call for tabulate_tensor on a cell
  const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                           int)>&
  get_tabulate_tensor_fn_cell(unsigned int i) const;

  /// Get the function for 'tabulate_tensor' for exterior facet integral i
  /// @param i
  ///    Integral number
  /// @returns std::function
  ///    Function to call for tabulate_tensor on an exterior facet
  const std::function<void(PetscScalar*, const PetscScalar*, const double*, int,
                           int)>&
  get_tabulate_tensor_fn_exterior_facet(unsigned int i) const;

  /// Get the function for 'tabulate_tensor' for interior facet integral i
  /// @param i
  ///    Integral number
  /// @returns std::function
  ///    Function to call for tabulate_tensor on an interior facet
  const std::function<void(PetscScalar*, const PetscScalar* w, const double*,
                           const double*, int, int, int, int)>&
  get_tabulate_tensor_fn_interior_facet(unsigned int i) const;

  /// Register the function for 'tabulate_tensor' for cell integral i
  void register_tabulate_tensor_cell(int i, void (*fn)(PetscScalar*,
                                                       const PetscScalar*,
                                                       const double*, int));

  /// Register the function for 'tabulate_tensor' for exterior facet integral
  /// i
  void register_tabulate_tensor_exterior_facet(
      int i,
      void (*fn)(PetscScalar*, const PetscScalar*, const double*, int, int));

  /// Register the function for 'tabulate_tensor' for exterior facet integral
  /// i
  void register_tabulate_tensor_interior_facet(
      int i, void (*fn)(PetscScalar*, const PetscScalar* w, const double*,
                        const double*, int, int, int, int));

  /// Number of integrals of given type
  int num_integrals(FormIntegrals::Type t) const;

  /// Get the integer IDs of integrals of type t.
  /// The IDs correspond to the domains which the integrals are defined for in
  /// the form, except ID -1, which denotes the default integral.
  const std::vector<int>& integral_ids(FormIntegrals::Type t) const;

  /// Get the list of active entities for the ith integral of type t.
  /// Note, these are not retrieved by ID, but stored in order. The IDs can
  /// be obtained with "FormIntegrals::integral_ids()"
  /// For cell integrals, a list of cells. For facet integrals, a list of facets
  /// etc.
  const std::vector<std::int32_t>& integral_domains(FormIntegrals::Type t,
                                                    unsigned int i) const;

  /// Set the valid domains for the integrals of a given type from a
  /// MeshFunction "marker". The MeshFunction should have a value for each cell
  /// (entity) which corresponds to an integral ID. Note the MeshFunction is not
  /// stored, so if there any changes to the integration domain this must be
  /// called again.
  void set_domains(FormIntegrals::Type type,
                   const mesh::MeshFunction<std::size_t>& marker);

  /// If there exists a default integral of any type, set the list of entities
  /// for those integrals from the mesh topology. For cell integrals, this is
  /// all cells. For facet integrals, it is either all interior or all exterior
  /// facets.
  void set_default_domains(const mesh::Mesh& mesh);

private:
  // Function pointers to tabulate_tensor functions
  std::vector<
      std::function<void(PetscScalar*, const PetscScalar*, const double*, int)>>
      _tabulate_tensor_cell;

  std::vector<std::function<void(PetscScalar*, const PetscScalar*,
                                 const double*, int, int)>>
      _tabulate_tensor_exterior_facet;

  std::vector<
      std::function<void(PetscScalar*, const PetscScalar* w, const double*,
                         const double*, int, int, int, int)>>
      _tabulate_tensor_interior_facet;

  // ID codes for each stored integral sorted numerically (-1 for default
  // integral is always first, if present) along with lists of entities which
  // are active for each integral.
  // FIXME: these could be consolidated
  std::vector<int> _cell_integral_ids;
  std::vector<std::vector<std::int32_t>> _cell_integral_domains;

  std::vector<int> _exterior_facet_integral_ids;
  std::vector<std::vector<std::int32_t>> _exterior_facet_integral_domains;

  std::vector<int> _interior_facet_integral_ids;
  std::vector<std::vector<std::int32_t>> _interior_facet_integral_domains;
};
} // namespace fem
} // namespace dolfin
