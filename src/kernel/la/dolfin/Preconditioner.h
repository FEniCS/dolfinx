// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PRECONDITIONER_H
#define __PRECONDITIONER_H

namespace dolfin
{

  class Vector;

  /// The class Preconditioner serves as a base class
  /// for all preconditioners, specifying the interface
  /// of a preconditioner.
  ///
  /// A preconditioner solves (approximately) a linear
  /// system Ax = b for a given right-hand side b. Note
  /// that the matrix A is not given as an argument to
  /// the function solve(), since a preconditioner may
  /// be defined independent of any matrix. However, a
  /// matrix is typically given to the constructor of
  /// the preconditioner.
  
  class Preconditioner
  {
  public:
    
    /// Constructor
    Preconditioner();

    /// Destructor
    virtual ~Preconditioner();

    /// Solve linear system Ax = b for given right-hand side b
    virtual void solve(Vector& x, const Vector& b) = 0;

  };

}

#endif
