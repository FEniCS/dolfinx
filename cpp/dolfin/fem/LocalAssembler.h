// Copyright (C) 2011 Marie E. Rognes
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;
class Facet;
template <typename T>
class MeshFunction;
} // namespace mesh

namespace fem
{
class UFC;

/// Assembly of local cell tensors. Used by the adaptivity and
/// LocalSolver functionality in dolfin. The local assembly
/// functionality provided here is also wrapped as a free function
/// assemble_local(form_a, cell) in Python for easier usage. Use
/// from the C++ interface defined below will be faster than the
/// free function as fewer objects need to be created and destroyed.

class LocalAssembler
{

public:
  /// Assemble a local tensor on a cell. Internally calls
  /// assemble_cell(), assemble_exterior_facet(),
  /// assemble_interior_facet().
  static void assemble(
      Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>& A, ///< [out] The tensor to assemble.
      UFC& ufc,                          ///< [in]
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs,     ///< [in]
      const mesh::Cell& cell,                                        ///< [in]
      const mesh::MeshFunction<std::size_t>* cell_domains,           ///< [in]
      const mesh::MeshFunction<std::size_t>* exterior_facet_domains, ///< [in]
      const mesh::MeshFunction<std::size_t>* interior_facet_domains  ///< [in]
  );

  /// Worker method called by assemble() to perform assembly of
  /// volume integrals (UFL measure dx).
  static void assemble_cell(
      Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>& A, ///< [out] The tensor to assemble.
      UFC& ufc,                          ///< [in]
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs, ///< [in]
      const mesh::Cell& cell,                                    ///< [in]
      const mesh::MeshFunction<std::size_t>* cell_domains        ///< [in]
  );

  /// Worker method called by assemble() for each of the cell's
  /// external facets to perform assembly of external facet
  /// integrals (UFL measure ds).
  static void assemble_exterior_facet(
      Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>& A, ///< [out] The tensor to assemble.
      UFC& ufc,                          ///< [in]
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs,    ///< [in]
      const mesh::Cell& cell,                                       ///< [in]
      const mesh::Facet& facet,                                     ///< [in]
      const std::size_t local_facet,                                ///< [in]
      const mesh::MeshFunction<std::size_t>* exterior_facet_domains ///< [in]
  );

  /// Worker method called by assemble() for each of the cell's
  /// internal facets to perform assembly of internal facet
  /// integrals (UFL measure dS)
  static void assemble_interior_facet(
      Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>& A, ///< [out] The tensor to assemble.
      UFC& ufc,                          ///< [in]
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs,     ///< [in]
      const mesh::Cell& cell,                                        ///< [in]
      const mesh::Facet& facet,                                      ///< [in]
      const std::size_t local_facet,                                 ///< [in]
      const mesh::MeshFunction<std::size_t>* interior_facet_domains, ///< [in]
      const mesh::MeshFunction<std::size_t>* cell_domains            ///< [in]
  );
};
} // namespace fem
} // namespace dolfin
