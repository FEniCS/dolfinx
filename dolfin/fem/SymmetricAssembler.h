// Copyright (C) 2012 Joachim B. Haga
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-01-26
// Last changed: 2012-10-04

#ifndef __SYMMETRIC_ASSEMBLER_H
#define __SYMMETRIC_ASSEMBLER_H

#include "Assembler.h"

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class DirichletBC;

  /// This class provides implements an assembler for systems
  /// of the form Ax = b. Its assembly algorithms are similar to SystemAssember's,
  /// but it saves the matrix modifications into a separate tensor so that it
  /// can later apply the symmetric modifications to a RHS vector.

  /// The non-symmetric part is only nonzero in BC columns, and is zero in all BC
  /// rows, so that [(A_s+A_n) x = b] implies [A_s x = b - A_n b], IF b has
  /// boundary conditions applied. (If the final A is composed from a sum of
  /// A_s matrices, BCs must be reapplied to make the diagonal value for BC
  /// dofs 1.0. The matrix will remain symmetric since only the diagonal is
  /// changed.)
  ///
  /// *Example*
  ///
  ///    .. code-block:: c++
  ///
  ///        std::vector<const DirichletBC*> bcs = {bc};
  ///        SymmetricAssembler::assemble(A, A_n, a, bcs, bcs);
  ///        Assembler::assemble(b, L);
  ///        bc.apply(b)
  ///        A_n.mult(b, b_mod);
  ///        b -= b_mod;

  class SymmetricAssembler : public Assembler
  {
  public:

    /// Constructor
    SymmetricAssembler() {}

    /// Assemble a and apply Dirichlet boundary conditions. Returns two
    /// matrices, where the second contains the symmetric modifications
    /// suitable for modifying RHS vectors.
    ///
    /// Note: row_bcs and col_bcs will normally be the same, but are different
    /// for e.g. off-diagonal block matrices in a mixed PDE.
    void assemble(GenericMatrix& A,
                  GenericMatrix& B,
                  const Form& a,
                  const std::vector<const DirichletBC*> row_bcs,
                  const std::vector<const DirichletBC*> col_bcs,
                  const MeshFunction<std::size_t>* cell_domains=NULL,
                  const MeshFunction<std::size_t>* exterior_facet_domains=NULL,
                  const MeshFunction<std::size_t>* interior_facet_domains=NULL);

    /// Assemble a and apply Dirichlet boundary conditions. Returns two
    /// matrices, where the second contains the symmetric modifications
    /// suitable for modifying RHS vectors.
    ///
    /// Note: row_bcs and col_bcs will normally be the same, but are different
    /// for e.g. off-diagonal block matrices in a mixed PDE.
    void assemble(GenericMatrix& A,
                  GenericMatrix& B,
                  const Form& a,
                  const std::vector<const DirichletBC*> row_bcs,
                  const std::vector<const DirichletBC*> col_bcs,
                  const SubDomain& sub_domain);

  protected:

    /// Add cell tensor to global tensor. Hook to allow the SymmetricAssembler
    /// to split the cell tensor into symmetric/antisymmetric parts.
    virtual void add_to_global_tensor(GenericTensor& A,
                                      std::vector<double>& local_A,
                                      std::vector<const std::vector<DolfinIndex>* >& dofs);

  private:

    // Parameters and methods used in add_to_global_tensor. The parameters
    // are stored (by the assemble() method) in this instance, since the
    // add_to_global_tensor interface is fixed.

    class Private;
    Private *impl;

  };

}

#endif
