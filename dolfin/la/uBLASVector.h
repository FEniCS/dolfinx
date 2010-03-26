// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Alnæs, 2008.
//
// First added:  2006-03-04
// Last changed: 2009-09-08

#ifndef __UBLAS_VECTOR_H
#define __UBLAS_VECTOR_H

#include <boost/shared_ptr.hpp>
#include "ublas.h"
#include "GenericVector.h"

namespace dolfin
{

  namespace ublas = boost::numeric::ublas;

  template<class T> class Array;

  /// This class provides a simple vector class based on uBLAS.
  /// It is a simple wrapper for a uBLAS vector implementing the
  /// GenericVector interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the underlying uBLAS vector and use the standard
  /// uBLAS interface which is documented at
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.

  class uBLASVector : public GenericVector
  {
  public:

    /// Create empty vector
    uBLASVector();

    /// Create vector of size N
    explicit uBLASVector(uint N);

    /// Copy constructor
    explicit uBLASVector(const uBLASVector& x);

    /// Construct vector from a ublas_vector
    explicit uBLASVector(boost::shared_ptr<ublas_vector> x);

    // Create vector from given uBLAS vector expression
    //template <class E>
    //explicit uBLASVector(const ublas::vector_expression<E>& x) : x(new ublas_vector(x)) {}

    /// Destructor
    virtual ~uBLASVector();

    //--- Implementation of the GenericTensor interface ---

    /// Create copy of tensor
    virtual uBLASVector* copy() const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericVector interface ---

    /// Resize vector to size N
    virtual void resize(uint N);

    /// Return size of vector
    virtual uint size() const;

    /// Return local ownership range of a vector
    virtual std::pair<uint, uint> local_range() const;

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows) const;

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows);

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows);

    /// Get all values on local process
    virtual void get_local(Array<double>& values) const;

    /// Set all values on local process
    virtual void set_local(const Array<double>& values);

    /// Add values to each entry on local process
    virtual void add_local(const Array<double>& values);

    /// Gather entries into local vector x
    virtual void gather(GenericVector& x, const Array<uint>& indices) const
    { not_working_in_parallel("uBLASVector::gather)"); }

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x);

    /// Return inner product with given vector
    virtual double inner(const GenericVector& x) const;

    /// Compute norm of vector
    virtual double norm(std::string norm_type) const;

    /// Return minimum value of vector
    virtual double min() const;

    /// Return maximum value of vector
    virtual double max() const;

    /// Return sum of values of vector
    virtual double sum() const;

    /// Multiply vector by given number
    virtual const uBLASVector& operator*= (double a);

    /// Multiply vector by another vector pointwise
    virtual const uBLASVector& operator*= (const GenericVector& x);

    /// Divide vector by given number
    virtual const uBLASVector& operator/= (double a);

    /// Add given vector
    virtual const uBLASVector& operator+= (const GenericVector& x);

    /// Subtract given vector
    virtual const uBLASVector& operator-= (const GenericVector& x);

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x);

    /// Assignment operator
    virtual const uBLASVector& operator= (double a);

    /// Return pointer to underlying data (const version)
    virtual const double* data() const
    { return &x->data()[0]; }

    /// Return pointer to underlying data
    virtual double* data()
    { return &x->data()[0]; }

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    //--- Special uBLAS functions ---

    /// Return reference to uBLAS vector (const version)
    const ublas_vector& vec() const
    { return *x; }

    /// Return reference to uBLAS vector (non-const version)
    ublas_vector& vec()
    { return *x; }

    /// Access value of given entry (const version)
    virtual double operator[] (uint i) const
    { return (*x)(i); };

    /// Access value of given entry (non-const version)
    double& operator[] (uint i)
    { return (*x)(i); };

    /// Assignment operator
    const uBLASVector& operator= (const uBLASVector& x);

  private:

    // Smart pointer to uBLAS vector object
    boost::shared_ptr<ublas_vector> x;

  };

}

#endif
