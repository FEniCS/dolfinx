// Copyright (C) 2004-2008 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2007.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Alnæs, 2008.
//
// First added:  2004-01-01
// Last changed: 2008-04-14

#ifndef __PETSC_VECTOR_H
#define __PETSC_VECTOR_H

#ifdef HAS_PETSC

#include <petscvec.h>

#include <dolfin/log/LogStream.h>
#include <dolfin/common/Variable.h>
#include "PETScObject.h"
#include "GenericVector.h"

namespace dolfin
{

  /// This class provides a simple vector class based on PETSc.
  /// It is a simple wrapper for a PETSc vector pointer (Vec)
  /// implementing the GenericVector interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Vec pointer using the function vec() and
  /// use the standard PETSc interface.

  class PETScVector : public GenericVector, public PETScObject, public Variable
  {
  public:

    /// Create empty vector
    explicit PETScVector();

    /// Create vector of size N
    explicit PETScVector(uint N);

    /// Copy constructor
    explicit PETScVector(const PETScVector& x);

    /// Create vector from given PETSc Vec pointer
    explicit PETScVector(Vec x);

    /// Destructor
    ~PETScVector();

    //--- Implementation of the GenericTensor interface ---

    /// Return copy of tensor
    PETScVector* copy() const;

    /// Set all entries to zero and keep any sparse structure
    void zero();

    /// Finalize assembly of tensor
    void apply();

    /// Display tensor
    void disp(uint precision=2) const;

    //--- Implementation of the GenericVector interface ---

    /// Initialize vector of size N
    void init(uint N);

    /// Return size of vector
    uint size() const;

    /// Get block of values
    void get(real* block, uint m, const uint* rows) const;

    /// Set block of values
    void set(const real* block, uint m, const uint* rows);

    /// Add block of values
    void add(const real* block, uint m, const uint* rows);

    /// Get all values
    void get(real* values) const;

    /// Set all values
    void set(real* values);

    /// Add values to each entry
    void add(real* values);

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(real a, const GenericVector& x); 

    /// Return inner product with given vector
    real inner(const GenericVector& v) const;

    /// Return norm of vector
    real norm(VectorNormType type=l2) const;

    /// Multiply vector by given number
    const PETScVector& operator*= (real a);

    /// Assignment operator
    const GenericVector& operator= (const GenericVector& x);

    /// Assignment operator
    const PETScVector& operator= (const PETScVector& x);

    //--- Convenience functions ---

    /// Add given vector
    const PETScVector& operator+= (const GenericVector& x);

    /// Subtract given vector
    const PETScVector& operator-= (const GenericVector& x);

    /// Divide vector by given number
    const PETScVector& operator/= (real a);

    //--- Special functions ---

    /// Return linear algebra backend factory
    LinearAlgebraFactory& factory() const;

    //--- Special PETScFunctions ---

    /// Return PETSc Vec pointer
    Vec vec() const;

    friend class PETScMatrix;

  private:

    // PETSc Vec pointer
    Vec x;
    
    // True if the pointer is a copy of someone else's data
    bool _copy;

  };

  /// Output of PETScVector
  LogStream& operator<< (LogStream& stream, const PETScVector& x);
  
}

#endif

#endif
