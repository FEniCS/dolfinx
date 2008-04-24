// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21
// Last changed:  2008-04-23

#ifndef __EPETRA_VECTOR_H
#define __EPETRA_VECTOR_H

#ifdef HAS_TRILINOS

/*
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_FEVector.h>
*/

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Variable.h>
#include "GenericVector.h"
#include "LinearAlgebraFactory.h"
#include "VectorNormType.h"

class Epetra_FEVector; 
class Epetra_Map;

namespace dolfin
{
  /// This class represents a vector of dimension N.
  /// It is a simple wrapper for a Epetra vector pointer (Epetra_FEVector).
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the Epetra_FEVector pointer using the function vec() and
  /// use the standard Epetra interface.

  class EpetraVector: public GenericVector, public Variable
  {
  public:

    /// Empty vector
    EpetraVector();

    /// Create vector of given size
    explicit EpetraVector(uint N);

    /// Create vector view from given Epetra_FEVector pointer
    explicit EpetraVector(Epetra_FEVector* vector);

    /// Create vector from given Epetra_Map reference
    explicit EpetraVector(const Epetra_Map& map);

    /// Copy constructor
    explicit EpetraVector(const EpetraVector& x);

    /// Destructor
    ~EpetraVector();

    /// Return backend factory
    LinearAlgebraFactory& factory() const;

    /// Initialize vector data
    void init(uint N);

    /// Create uninitialized vector
    EpetraVector* create() const;

    /// Create copy of vector
    EpetraVector* copy() const;

    /// Return size of vector
    uint size() const;

    /// Set all entries to zero
    void zero();

    /// Assignment of vector
    const EpetraVector& operator= (const GenericVector& x);

    /// Assignment of vector
    const EpetraVector& operator= (const EpetraVector& x);

    /// Add vector x
    const EpetraVector& operator+= (const GenericVector& x);

    /// Subtract vector x
    const EpetraVector& operator-= (const GenericVector& x);

    /// Multiply vector with scalar
    const EpetraVector& operator*= (real a);

    /// Divide vector by given number
    virtual const EpetraVector& operator/= (real a)
    { *this *= 1.0 / a; return *this; }

    /// Apply changes to vector
    void apply();

    /// Display vector
    void disp(uint precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const EpetraVector& A);

    // Friends
    friend class EpetraMatrix;

    /// Get values
    void get(real* values) const;

    /// Set values
    void set(real* values);

    /// Add values
    void add(real* values);

    /// Get block of values
    void get(real* block, uint m, const uint* rows) const;

    /// Set block of values
    void set(const real* block, uint m, const uint* rows);

    /// Add block of values
    void add(const real* block, uint m, const uint* rows);

    /// Return Epetra_MultiVector reference
    Epetra_FEVector& vec() const;

    /// Inner product 
    virtual real inner(const GenericVector& vector) const; 

    //  this += a*x   
    virtual void axpy(real a, const GenericVector& x); 

    /// Return norm of vector
    virtual real norm(VectorNormType type = l2) const { 
      error("EpetraVector::norm Not implemented yet"); 
      return -1.0; 
    }

  private:

    // Epetra_FEVector pointer
    Epetra_FEVector* x;
    
    // True if the pointer is a copy of someone else's data
    bool _copy;
    
  };  

  LogStream& operator<< (LogStream& stream, const EpetraVector& A);

}

#endif
#endif
