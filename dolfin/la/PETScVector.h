// Copyright (C) 2004-2008 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2007.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Alnæs, 2008.
//
// First added:  2004-01-01
// Last changed: 2008-04-29


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
    PETScVector();

    /// Create vector of size N
    explicit PETScVector(uint N);

    /// Copy constructor
    explicit PETScVector(const PETScVector& x);

    /// Create vector from given PETSc Vec pointer
    explicit PETScVector(Vec x);

    /// Destructor
    virtual ~PETScVector();

    //--- Implementation of the GenericTensor interface ---

    /// Return copy of tensor
    virtual PETScVector* copy() const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply();

    /// Display tensor
    virtual void disp(uint precision=2) const;

    //--- Implementation of the GenericVector interface ---

    /// Initialize vector of size N
    virtual void init(uint N);

    /// Return size of vector
    virtual uint size() const;

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows) const;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows);

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows);

    /// Get all values
    virtual void get(real* values) const;

    /// Set all values
    virtual void set(real* values);

    /// Add values to each entry
    virtual void add(real* values);

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(real a, const GenericVector& x); 

    /// Return inner product with given vector
    virtual real inner(const GenericVector& v) const;

    /// Return norm of vector
    virtual real norm(VectorNormType type=l2) const;

    /// Return minimum value of vector
    virtual real min() const;

    /// Return maximum value of vector
    virtual real max() const;

    /// Multiply vector by given number
    virtual const PETScVector& operator*= (real a);

    /// Divide vector by given number
    virtual const PETScVector& operator/= (real a);

    /// Add given vector
    virtual const PETScVector& operator+= (const GenericVector& x);

    /// Subtract given vector
    virtual const PETScVector& operator-= (const GenericVector& x);

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x);

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    //--- Special PETSc functions ---

    /// Return PETSc Vec pointer
    Vec vec() const;

    /// Assignment operator
    const PETScVector& operator= (const PETScVector& x);

    friend class PETScMatrix;

  private:

    // PETSc Vec pointer
    Vec x;
    
    // True if we don't own the vector x points to
    bool is_view;

  };

  /// Output of PETScVector
  LogStream& operator<< (LogStream& stream, const PETScVector& x);
  
}

#endif

#endif
