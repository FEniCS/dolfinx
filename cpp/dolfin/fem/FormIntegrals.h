// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <petscsys.h>
#include <vector>

namespace dolfin
{
namespace mesh
{
class Mesh;
template <typename T>
class MeshFunction;
} // namespace mesh

namespace fem
{

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

  /// Construct empty object
  FormIntegrals();

  /// Get the function for 'tabulate_tensor' for integral i of given
  /// type
  /// @param[in] type Integral type
  /// @param[in] i Integral number
  /// @return Function to call for tabulate_tensor
  const std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*, const double*,
                           const int*, const int*)>&
  get_tabulate_tensor_function(FormIntegrals::Type type, unsigned int i) const;

  /// Register the function for 'tabulate_tensor' for integral i of
  /// given type
  void register_tabulate_tensor(FormIntegrals::Type type, int i,
                                void (*fn)(PetscScalar*, const PetscScalar*,
                                           const PetscScalar*,
                                           const double*, const int*,
                                           const int*));

  /// Number of integrals of given type
  int num_integrals(FormIntegrals::Type t) const;

  /// Get the integer IDs of integrals of type t. The IDs correspond to
  /// the domains which the integrals are defined for in the form,
  /// except ID -1, which denotes the default integral.
  std::vector<int> integral_ids(FormIntegrals::Type t) const;

  /// Get the list of active entities for the ith integral of type t.
  /// Note, these are not retrieved by ID, but stored in order. The IDs
  /// can be obtained with "FormIntegrals::integral_ids()". For cell
  /// integrals, a list of cells. For facet integrals, a list of facets
  /// etc.
  const std::vector<std::int32_t>& integral_domains(FormIntegrals::Type t,
                                                    unsigned int i) const;

  /// Set the valid domains for the integrals of a given type from a
  /// MeshFunction "marker". The MeshFunction should have a value for
  /// each cell (entity) which corresponds to an integral ID. Note the
  /// MeshFunction is not stored, so if there any changes to the
  /// integration domain this must be called again.
  void set_domains(FormIntegrals::Type type,
                   const mesh::MeshFunction<std::size_t>& marker);

  /// If there exists a default integral of any type, set the list of
  /// entities for those integrals from the mesh topology. For cell
  /// integrals, this is all cells. For facet integrals, it is either
  /// all interior or all exterior facets.
  void set_default_domains(const mesh::Mesh& mesh);

private:
  // Collect together the function, id, and indices of entities to
  // integrate on
  struct Integral
  {
    std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*, const double*,
                       const int*, const int*)>
        tabulate;
    int id;
    std::vector<std::int32_t> active_entities;
  };

  // Array of vectors of integrals, arranged by type (see Type enum, and
  // struct Integral above)
  std::array<std::vector<struct Integral>, 4> _integrals;
};
} // namespace fem
} // namespace dolfin
