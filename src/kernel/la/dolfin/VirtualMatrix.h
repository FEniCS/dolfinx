// Copyright (C) 2003 Hoffman/Logg and Erik Svensson.
// Licensed under the GNU GPL Version 2.

#ifndef __VIRTUAL_MATRIX_H
#define __VIRTUAL_MATRIX_H

namespace dolfin {

  /// A VirtualMatrix represents a matrix and can be used together with a KrylovSolver
  /// for systems where for some reason the matrix is not available.
  class VirtualMatrix {
  public:

    /// Multiply x with the matrix and put the result in Ax
    virtual void mult(const Vector& x, Vector& Ax) {};

    /// Solve (approximately) the system A x = b
    virtual void precondition(Vector& x, const Vector& b) {};

  };
  
}

#endif
