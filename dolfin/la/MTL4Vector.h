// Copyright (C) 2008 Dag Lindbo
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
// Modified by Anders Logg, 2008-2010.
// Modified by Garth N. Wells, 2009.
//
// First added:  2008-07-06
// Last changed: 2011-01-14

#ifdef HAS_MTL4

#ifndef __MTL4_VECTOR_H
#define __MTL4_VECTOR_H

#include <vector>
#include <utility>
#include <string>

#include <dolfin/common/types.h>
#include "mtl4.h"
#include "GenericVector.h"

//  Developers note:
//
//  This class implements a minimal backend for MTL4.
//
//  There are certain inline decisions that have been deferred.
//  Due to the extensive calling of this backend through the generic LA
//  interface, it is not clear where inlining will be possible and
//  improve performance.

namespace dolfin
{

  template<typename T> class Array;

  class MTL4Vector: public GenericVector
  {
  public:

    /// Create empty vector
    MTL4Vector();

    /// Create vector of size N
    explicit MTL4Vector(uint N);

    /// Copy constructor
    MTL4Vector(const MTL4Vector& x);

    /// Destructor
    virtual ~MTL4Vector();

    //--- Implementation of the GenericTensor interface ---

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericVector interface ---

    /// Return copy of vector
    virtual boost::shared_ptr<GenericVector> copy() const;

    /// Resize vector to size N
    virtual void resize(uint N);

    /// Resize vector with given ownership range
    virtual void resize(std::pair<uint, uint> range);

    /// Resize vector with given ownership range and with ghost values
    virtual void resize(std::pair<uint, uint> range,
                        const std::vector<uint>& ghost_indices);

    /// Return true if vector is empty
    virtual bool empty() const;

    /// Return size of vector
    virtual uint size() const;

    /// Return local size of vector
    virtual uint local_size() const
    { return size(); }

    /// Return local ownership range of a vector
    virtual std::pair<uint, uint> local_range() const;

    /// Determine whether global vector index is owned by this process
    virtual bool owns_index(uint i) const;

    /// Get block of values
    virtual void get_local(double* block, uint m, const uint* rows) const;

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows);

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows);

    /// Get all values on local process
    virtual void get_local(std::vector<double>& values) const;

    /// Set all values on local process
    virtual void set_local(const std::vector<double>& values);

    /// Add all values to each entry on local process
    virtual void add_local(const Array<double>& values);

    /// Gather entries into local vector x
    virtual void gather(GenericVector& x, const std::vector<uint>& indices) const;

    /// Gather entries into x
    virtual void gather(std::vector<double>& x, const std::vector<uint>& indices) const;

    /// Gather all entries into x on process 0
    virtual void gather_on_zero(std::vector<double>& x) const;

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x);

    /// Replace all entries in the vector by their absolute values
    virtual void abs();

    /// Return inner product with given vector
    virtual double inner(const GenericVector& vector) const;

    /// Return norm of vector
    virtual double norm(std::string norm_type) const;

    /// Return minimum value of vector
    virtual double min() const;

    /// Return maximum value of vector
    virtual double max() const;

    /// Return sum of values of vector
    virtual double sum() const;

    /// Return sum of selected rows in vector. Repeated entries are only summed once.
    virtual double sum(const Array<uint>& rows) const;

    /// Multiply vector by given number
    virtual const MTL4Vector& operator*= (double a);

    /// Multiply vector by another vector pointwise
    virtual const MTL4Vector& operator*= (const GenericVector& x);

    /// Divide vector by given number
    virtual const MTL4Vector& operator/= (double a);

    /// Assignment operator
    virtual const MTL4Vector& operator= (double a);

    /// Add given vector
    virtual const MTL4Vector& operator+= (const GenericVector& x);

    /// Add number to all components of a vector
    virtual const MTL4Vector& operator+= (double a);

    /// Subtract given vector
    virtual const MTL4Vector& operator-= (const GenericVector& x);

    /// Subtract number from all components of a vector
    virtual const MTL4Vector& operator-= (double a);

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x);

    /// Return pointer to underlying data (const version)
    virtual const double* data() const
    { return x.address_data(); }

    /// Return pointer to underlying data (non-const version)
    virtual double* data()
    { return x.address_data(); }

    //--- Special functions ---
    virtual GenericLinearAlgebraFactory& factory() const;

    //--- Special MTL4 functions ---

    /// Return const mtl4_vector reference
    const mtl4_vector& vec() const;

    /// Return mtl4_vector reference
    mtl4_vector& vec();

    /// Assignment operator
    virtual const MTL4Vector& operator= (const MTL4Vector& x);

  private:

    // MTL4 vector object
    mtl4_vector x;

  };

}

#endif
#endif
