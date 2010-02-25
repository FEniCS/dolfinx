// Copyright (C) 2008 Kent-Andre Mardal
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS

#ifndef __EPETRA_USER_PRECONDITIONER_SOLVER_H
#define __EPETRA_USER_PRECONDITIONER_SOLVER_H

#include <string>

namespace dolfin
{
  class EpetraVector;
  class EpetraMatrix;

  /// This class specifies the interface for user-defined Krylov
  /// method EpetraUserPreconditioner. A user wishing to implement her own
  /// EpetraUserPreconditioner needs only supply a function that approximately
  /// solves the linear system given a right-hand side.

  class EpetraUserPreconditioner
  {
  public:
    /// Constructor
    EpetraUserPreconditioner() {};

    /// Destructor
    virtual ~EpetraUserPreconditioner() {};

    /// Set the Preconditioner type (amg, ilu, etc.)
    void set_type(std::string type);

    /// Initialise preconditioner
    virtual void init(const EpetraMatrix& A);

    /// Solve linear system (M^-1)Ax = y
    virtual void solve(EpetraVector& x, const EpetraVector& b);

  private:

    // Preconditioner type
    std::string type;

  };

}

#endif
#endif


