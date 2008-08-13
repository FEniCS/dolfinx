// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-06
// Last changed: 2008-07-06

#ifdef HAS_MTL4

#ifndef __MTL4_VECTOR_H
#define __MTL4_VECTOR_H

#include <dolfin/log/LogStream.h>
#include <dolfin/common/Variable.h>
#include "mtl4.h"
#include "GenericVector.h"

/*
  Developers note:
  
  This class implements a minimal backend for MTL4.

  There are certain inline decisions that have been deferred.
  Due to the extensive calling of this backend through the generic LA
  interface, it is not clear where inlining will be possible and 
  improve performance.
*/

namespace dolfin
{

  class MTL4Vector: public GenericVector, public Variable
  {
  public:

    /// Create empty vector
    MTL4Vector();

    /// Create vector of size N
    explicit MTL4Vector(uint N);

    /// Copy constructor
    explicit MTL4Vector(const MTL4Vector& x);

    /// Destructor
    virtual ~MTL4Vector();

    //--- Implementation of the GenericTensor interface ---

    /// Return copy of tensor
    virtual MTL4Vector* copy() const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply();

    /// Display vector
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

    /// Add all values to each entry
    virtual void add(real* values);

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(real a, const GenericVector& x);

    /// Return inner product with given vector
    virtual real inner(const GenericVector& vector) const;

    /// Return norm of vector
    virtual real norm(VectorNormType type = l2) const;

    /// Return minimum value of vector
    virtual real min() const;

    /// Return maximum value of vector
    virtual real max() const;

    /// Multiply vector by given number
    virtual const MTL4Vector& operator*= (real a);

    /// Divide vector by given number
    virtual const MTL4Vector& operator/= (real a);

    /// Assignment operator
    virtual const MTL4Vector& operator= (real a);

    /// Add given vector
    virtual const MTL4Vector& operator+= (const GenericVector& x);

    /// Subtract given vector
    virtual const MTL4Vector& operator-= (const GenericVector& x);

    /// Assignment operator
    virtual const MTL4Vector& operator= (const GenericVector& x);

    /// Return pointer to underlying data (const version)
    virtual const real* data() const
    { return x.address_data(); }

    /// Return pointer to underlying data (const version)
    virtual real* data()
    { return x.address_data(); }

    //--- Special functions ---
    virtual LinearAlgebraFactory& factory() const;

    //--- Special MTL4 functions ---

    /// Return const mtl4_vector reference
    const mtl4_vector& vec() const;
    
    /// Return mtl4_vector reference
    mtl4_vector& vec();

    /// Assignment operator
    const MTL4Vector& operator= (const MTL4Vector& x);

    //friend class MTL4Matrix;

  private:

    // MTL4 vector object
    mtl4_vector x;

  };  

  LogStream& operator<< (LogStream& stream, const MTL4Vector& A);

}

#endif 
#endif 
