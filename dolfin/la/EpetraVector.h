// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-04-21
// Last changed: 2008-04-25

#ifndef __EPETRA_VECTOR_H
#define __EPETRA_VECTOR_H

#ifdef HAS_TRILINOS

#include <dolfin/log/LogStream.h>
#include <dolfin/common/Variable.h>
#include "GenericVector.h"

class Epetra_FEVector;
class Epetra_Map;

namespace dolfin
{

  /// This class provides a simple vector class based on Epetra.
  /// It is a simple wrapper for an Epetra vector object (Epetre_FEVector)
  /// implementing the GenericVector interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the Epetra_FEVector object using the function vec() and
  /// use the standard Epetra interface.

  class EpetraVector: public GenericVector, public Variable
  {
  public:

    /// Create empty vector
    explicit EpetraVector();

    /// Create vector of size N
    explicit EpetraVector(uint N);

    /// Copy constructor
    explicit EpetraVector(const EpetraVector& x);

    /// Create vector view from given Epetra_FEVector pointer
    explicit EpetraVector(Epetra_FEVector* vector);

    /// Create vector from given Epetra_Map
    explicit EpetraVector(const Epetra_Map& map);

    /// Destructor
    ~EpetraVector();

    //--- Implementation of the GenericTensor interface ---

    /// Return copy of tensor
    EpetraVector* copy() const;

    /// Set all entries to zero and keep any sparse structure
    void zero();

    /// Finalize assembly of tensor
    void apply();

    /// Display vector
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

    /// Add all values to each entry
    void add(real* values);

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(real a, const GenericVector& x);

    /// Return inner product with given vector
    virtual real inner(const GenericVector& vector) const;

    /// Return norm of vector
    virtual real norm(VectorNormType type = l2) const;

    /// Multiply vector by given number
    const EpetraVector& operator*= (real a);

    /// Divide vector by given number
    virtual const EpetraVector& operator/= (real a)
    { *this *= 1.0 / a; return *this; }

    /// Add given vector
    const EpetraVector& operator+= (const GenericVector& x);

    /// Subtract given vector
    const EpetraVector& operator-= (const GenericVector& x);

    /// Assignment operator
    const EpetraVector& operator= (const GenericVector& x);

    /// Assignment operator 
    const EpetraVector& operator= (const EpetraVector& x);

    //--- Special functions ---

    /// Return linear algebra backend factory
    LinearAlgebraFactory& factory() const;

    //--- Special Epetra functions ---

    /// Return Epetra_FEVector reference
    Epetra_FEVector& vec() const;

    /// Create uninitialized vector
    EpetraVector* create() const;

    friend class EpetraMatrix;

  private:

    // Epetra_FEVector pointer
    Epetra_FEVector* x;

    // True if the pointer is a copy of someone else's data
    bool _copy;

  };  

  LogStream& operator<< (LogStream& stream, const EpetraVector& A);

}

#endif //HAS_TRILINOS
#endif //__EPETRA_VECTOR_H
