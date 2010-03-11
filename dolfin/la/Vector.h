// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2007-07-03
// Last changed: 2009-09-08

#ifndef __VECTOR_H
#define __VECTOR_H

#include "DefaultFactory.h"
#include "GenericVector.h"

namespace dolfin
{

  template<class T> class Array;

  /// This class provides the default DOLFIN vector class,
  /// based on the default DOLFIN linear algebra backend.

  class Vector : public GenericVector
  {
  public:

    /// Create empty vector
    Vector() : vector(0)
    { DefaultFactory factory; vector = factory.create_vector(); }

    /// Create vector of size N
    explicit Vector(uint N) : vector(0)
    { DefaultFactory factory; vector = factory.create_vector(); vector->resize(N); }

    /// Copy constructor
    explicit Vector(const Vector& x) : vector(x.vector->copy())
    {}

    /// Create a Vector from a GenericVetor
    explicit Vector(const GenericVector& x) : vector(x.factory().create_vector())
    { vector = x.copy(); }

    /// Destructor
    virtual ~Vector()
    { delete vector; }

    //--- Implementation of the GenericTensor interface ---

    /// Return copy of tensor
    virtual Vector* copy() const
    { Vector* x = new Vector(); delete x->vector; x->vector = vector->copy(); return x; }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    { vector->zero(); }

    /// Finalize assembly of tensor
    virtual void apply(std::string mode)
    { vector->apply(mode); }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const
    { return vector->str(verbose); }

    //--- Implementation of the GenericVector interface ---

    /// Resize vector to size N
    virtual void resize(uint N)
    { vector->resize(N); }

    /// Return size of vector
    virtual uint size() const
    { return vector->size(); }

    /// Return local ownership range of a vector
    virtual std::pair<uint, uint> local_range() const
    { return vector->local_range(); }

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows) const
    { vector->get(block, m, rows); }

    /// Get block of values (values must all live on the local process)
    virtual void get_local(double* block, uint m, const uint* rows) const
    { vector->get_local(block,m,rows); }

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows)
    { vector->set(block, m, rows); }

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows)
    { vector->add(block, m, rows); }

    /// Get all values on local process
    virtual void get_local(Array<double>& values) const
    { vector->get_local(values); }

    /// Set all values on local process
    virtual void set_local(const Array<double>& values)
    { vector->set_local(values); }

    /// Add values to each entry on local process
    virtual void add_local(const Array<double>& values)
    { vector->add_local(values); }

    /// Gather entries into local vector x
    virtual void gather(GenericVector& x, const Array<uint>& indices) const
    { vector->gather(x, indices); }

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x)
    { vector->axpy(a, x); }

    /// Return inner product with given vector
    virtual double inner(const GenericVector& x) const
    { return vector->inner(x); }

    /// Return norm of vector
    virtual double norm(std::string norm_type) const
    { return vector->norm(norm_type); }

    /// Return minimum value of vector
    virtual double min() const
    { return vector->min(); }

    /// Return maximum value of vector
    virtual double max() const
    { return vector->max(); }

    /// Return sum of values of vector
    virtual double sum() const
    { return vector->sum(); }

    /// Multiply vector by given number
    virtual const Vector& operator*= (double a)
    { *vector *= a; return *this; }

    /// Multiply vector by another vector pointwise
    virtual const Vector& operator*= (const GenericVector& x)
    { *vector *= x; return *this; }

    /// Divide vector by given number
    virtual const Vector& operator/= (double a)
    { *this *= 1.0 / a; return *this; }

    /// Add given vector
    virtual const Vector& operator+= (const GenericVector& x)
    { axpy(1.0, x); return *this; }

    /// Subtract given vector
    virtual const Vector& operator-= (const GenericVector& x)
    { axpy(-1.0, x); return *this; }

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x)
    { *vector = x; return *this; }

    /// Assignment operator
    const Vector& operator= (double a)
    { *vector = a; return *this; }

    /// Return pointer to underlying data (const version)
    virtual const double* data() const
    { return vector->data(); }

    /// Return pointer to underlying data
    virtual double* data()
    { return vector->data(); }

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const
    { return vector->factory(); }

    //--- Special functions, intended for library use only ---

    /// Return concrete instance / unwrap (const version)
    virtual const GenericVector* instance() const
    { return vector; }

    /// Return concrete instance / unwrap (non-const version)
    virtual GenericVector* instance()
    { return vector; }

    //--- Special Vector functions ---

    /// Assignment operator
    const Vector& operator= (const Vector& x)
    { *vector = *x.vector; return *this; }

  private:

    // Pointer to concrete implementation
    GenericVector* vector;

  };

}

#endif
