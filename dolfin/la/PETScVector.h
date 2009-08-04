// Copyright (C) 2004-2008 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2009.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Alnæs, 2008.
//
// First added:  2004-01-01
// Last changed: 2009-05-22

#ifndef __PETSC_VECTOR_H
#define __PETSC_VECTOR_H

#ifdef HAS_PETSC

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>
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

    /// Create vector of size N
    //PETScVector(uint N, std::string type = "global");

    /// Copy constructor
    PETScVector(const PETScVector& x);

    /// Create vector from given PETSc Vec pointer
    explicit PETScVector(boost::shared_ptr<Vec> x);

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

    /// Resize vector ro size N
    virtual void resize(uint N);

    /// Return size of vector
    virtual uint size() const;

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows) const;

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows);

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows);

    /// Get all values
    virtual void get(double* values) const;

    /// Set all values
    virtual void set(double* values);

    /// Add values to each entry
    virtual void add(double* values);

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x);

    /// Return inner product with given vector
    virtual double inner(const GenericVector& v) const;

    /// Return norm of vector
    virtual double norm(std::string norm_type = "l2") const;

    /// Return minimum value of vector
    virtual double min() const;

    /// Return maximum value of vector
    virtual double max() const;

    /// Return sum of values of vector
    virtual double sum() const;

    /// Multiply vector by given number
    virtual const PETScVector& operator*= (double a);

    /// Multiply vector by another vector pointwise
    virtual const PETScVector& operator*= (const GenericVector& x);

    /// Divide vector by given number
    virtual const PETScVector& operator/= (double a);

    /// Add given vector
    virtual const PETScVector& operator+= (const GenericVector& x);

    /// Subtract given vector
    virtual const PETScVector& operator-= (const GenericVector& x);

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x);

    /// Assignment operator
    virtual const PETScVector& operator= (double a);

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    //--- Special PETSc functions ---

    /// Return shared_ptr to PETSc Vec object
    boost::shared_ptr<Vec> vec() const;

    /// Assignment operator
    const PETScVector& operator= (const PETScVector& x);

    /// Gather vector entries into a local vector. If local_indices = 0, then
    /// a local index array is created such that the order of the values in the
    /// return array is the same as the order in global_indices.
    PETScVector gather(const uint* global_indices, const uint* local_indices, 
                       uint num_indices) const;

    friend class PETScMatrix;

  private:

    // Initialise PETSc vector
    void init(uint N, uint n, std::string type);

    // PETSc Vec pointer
    boost::shared_ptr<Vec> x;

    // PETSc norm types
    static const std::map<std::string, NormType> norm_types;

  };

  /// Output of PETScVector
  LogStream& operator<< (LogStream& stream, const PETScVector& x);

}

#endif

#endif
