// Copyright (C) 2004-2010 Johan Hoffman, Johan Jansson and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2005-2010.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Martin Alnæs, 2008.
//
// First added:  2004-01-01
// Last changed: 2011-01-14

#ifndef __PETSC_VECTOR_H
#define __PETSC_VECTOR_H

#ifdef HAS_PETSC

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

#include <petscvec.h>

#include "PETScObject.h"
#include "GenericVector.h"

#include <dolfin/log/dolfin_log.h>

namespace dolfin
{

  class PETScVectorDeleter
  {
  public:
    void operator() (Vec* x)
    {
      if (*x)
        VecDestroy(*x);
      delete x;
    }
  };

  class GenericSparsityPattern;
  template<class T> class Array;

  /// This class provides a simple vector class based on PETSc.
  /// It is a simple wrapper for a PETSc vector pointer (Vec)
  /// implementing the GenericVector interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Vec pointer using the function vec() and
  /// use the standard PETSc interface.

  class PETScVector : public GenericVector, public PETScObject
  {
  public:

    /// Create empty vector
    explicit PETScVector(std::string type="global");

    /// Create vector of size N
    PETScVector(uint N, std::string type="global");

    /// Create vector
    PETScVector(const GenericSparsityPattern& sparsity_pattern);

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
    virtual void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericVector interface ---

    /// Resize vector to global size N
    virtual void resize(uint N);

    /// Resize vector with given ownership range
    virtual void resize(std::pair<uint, uint> range);

    /// Resize vector with given ownership range and with ghost values
    virtual void resize(std::pair<uint, uint> range,
                        const std::vector<uint>& ghost_indices);

    /// Return size of vector
    virtual uint size() const;

    /// Return local size of vector
    virtual uint local_size() const;

    /// Return ownership range of a vector
    virtual std::pair<uint, uint> local_range() const;

    /// Determine whether global vector index is owned by this process
    virtual bool owns_index(uint i) const;

    /// Get block of values (values must all live on the local process)
    virtual void get_local(double* block, uint m, const uint* rows) const;

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

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x);

    /// Replace all entries in the vector by their absolute values
    virtual void abs();

    /// Return inner product with given vector
    virtual double inner(const GenericVector& v) const;

    /// Return norm of vector
    virtual double norm(std::string norm_type) const;

    /// Return minimum value of vector
    virtual double min() const;

    /// Return maximum value of vector
    virtual double max() const;

    /// Return sum of values of vector
    virtual double sum() const;

    /// Return sum of selected rows in vector
    virtual double sum(const Array<uint>& rows) const;

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

    virtual void update_ghost_values();

    //--- Special functions ---

    /// Reset data and PETSc vector object
    void reset();

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    //--- Special PETSc functions ---

    /// Return shared_ptr to PETSc Vec object
    boost::shared_ptr<Vec> vec() const;

    /// Assignment operator
    const PETScVector& operator= (const PETScVector& x);

    /// Gather vector entries into a local vector
    virtual void gather(GenericVector& y, const Array<uint>& indices) const;

    /// Gather entries into Array x
    virtual void gather(Array<double>& x, const Array<uint>& indices) const;

    // Test vector type (distributed/local)
    bool distributed() const;

    friend class PETScBaseMatrix;
    friend class PETScMatrix;

  private:

    // Initialise PETSc vector
    void init(std::pair<uint, uint> range, const std::vector<uint>& ghost_indices,
              bool distributed);

    // PETSc Vec pointer
    boost::shared_ptr<Vec> x;

    // PETSc Vec pointer (local ghosted)
    mutable boost::shared_ptr<Vec> x_ghosted;

    // Global-to-local map for ghost values
    boost::unordered_map<uint, uint> ghost_global_to_local;

    // PETSc norm types
    static const std::map<std::string, NormType> norm_types;

  };

}

#endif

#endif
