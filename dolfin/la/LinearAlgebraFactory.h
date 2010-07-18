// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-12-06

#ifndef __LINEAR_ALGEBRA_FACTORY_H
#define __LINEAR_ALGEBRA_FACTORY_H

namespace dolfin
{

  class GenericLinearSolver;
  class GenericMatrix;
  class GenericSparsityPattern;
  class GenericVector;

  class LinearAlgebraFactory
  {
    public:

    /// Constructor
    LinearAlgebraFactory() {}

    /// Destructor
    virtual ~LinearAlgebraFactory() {}

    /// Create empty matrix
    virtual dolfin::GenericMatrix* create_matrix() const = 0;

    /// Create empty vector (global)
    virtual dolfin::GenericVector* create_vector() const = 0;

    /// Create empty vector (local)
    virtual dolfin::GenericVector* create_local_vector() const = 0;

    /// Create empty sparsity pattern (returning zero if not used/needed)
    virtual dolfin::GenericSparsityPattern* create_pattern() const = 0;

    /// Create LU solver
    virtual dolfin::GenericLinearSolver* create_lu_solver() const = 0;

    /// Create Krylov solver
    virtual dolfin::GenericLinearSolver*
        create_krylov_solver(std::string method, std::string pc) const = 0;

  };

}

#endif
