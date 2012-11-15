// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring
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
// Modified by Garth N. Wells, 2008-2009.
//
// First added:  2008-04-21
// Last changed: 2011-01-14

#ifndef __EPETRA_VECTOR_H
#define __EPETRA_VECTOR_H

#ifdef HAS_TRILINOS

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <dolfin/common/types.h>
#include "GenericVector.h"

class Epetra_FEVector;
class Epetra_MultiVector;
class Epetra_Vector;
class Epetra_BlockMap;

namespace dolfin
{

  template <typename T> class Array;
  class GenericVector;

  /// This class provides a simple vector class based on Epetra.
  /// It is a simple wrapper for an Epetra vector object (Epetra_FEVector)
  /// implementing the GenericVector interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the Epetra_FEVector object using the function vec() or vec_ptr()
  /// and use the standard Epetra interface.

  class EpetraVector: public GenericVector
  {
  public:

    /// Create empty vector
    EpetraVector(std::string type="global");

    /// Create vector of size N
    explicit EpetraVector(std::size_t N, std::string type="global");

    /// Copy constructor
    EpetraVector(const EpetraVector& x);

    /// Create vector view from given Epetra_FEVector pointer
    explicit EpetraVector(boost::shared_ptr<Epetra_FEVector> vector);

    /// Create vector from given Epetra_BlockMap
    explicit EpetraVector(const Epetra_BlockMap& map);

    /// Destructor
    virtual ~EpetraVector();

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
    virtual void resize(std::size_t N);

    /// Resize vector with given ownership range
    virtual void resize(std::pair<std::size_t, std::size_t> range);

    /// Resize vector with given ownership range and with ghost values
    virtual void resize(std::pair<std::size_t, std::size_t> range,
                        const std::vector<std::size_t>& ghost_indices);

    /// Return true if vector is empty
    virtual bool empty() const;

    /// Return size of vector
    virtual std::size_t size() const;

    /// Return size of local vector
    virtual std::size_t local_size() const;

    /// Return local ownership range of a vector
    virtual std::pair<std::size_t, std::size_t> local_range() const;

    /// Determine whether global vector index is owned by this process
    virtual bool owns_index(std::size_t i) const;

    /// Set block of values
    virtual void set(const double* block, std::size_t m, const DolfinIndex* rows);

    /// Add block of values
    virtual void add(const double* block, std::size_t m, const DolfinIndex* rows);

    virtual void get_local(double* block, std::size_t m, const DolfinIndex* rows) const;

    /// Get all values on local process
    virtual void get_local(std::vector<double>& values) const;

    /// Set all values on local process
    virtual void set_local(const std::vector<double>& values);

    /// Add all values to each entry on local process
    virtual void add_local(const Array<double>& values);

    /// Gather entries into local vector x
    virtual void gather(GenericVector& x, const std::vector<DolfinIndex>& indices) const;

    /// Gather entries into x
    virtual void gather(std::vector<double>& x, const std::vector<DolfinIndex>& indices) const;

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

    /// Return sum of selected rows in vector
    virtual double sum(const Array<DolfinIndex>& rows) const;

    /// Multiply vector by given number
    virtual const EpetraVector& operator*= (double a);

    /// Multiply vector by another vector pointwise
    virtual const EpetraVector& operator*= (const GenericVector& x);

    /// Divide vector by given number
    virtual const EpetraVector& operator/= (double a);

    /// Add given vector
    virtual const EpetraVector& operator+= (const GenericVector& x);

    /// Add number to all components of a vector
    virtual const EpetraVector& operator+= (double a);

    /// Subtract given vector
    virtual const EpetraVector& operator-= (const GenericVector& x);

    /// Subtract number from all components of a vector
    virtual const EpetraVector& operator-= (double a);

    /// Assignment operator
    virtual const EpetraVector& operator= (const GenericVector& x);

    /// Assignment operator
    virtual const EpetraVector& operator= (double a);

    virtual void update_ghost_values();

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const;

    //--- Special Epetra functions ---

    /// Reset Epetra_FEVector
    void reset(const Epetra_BlockMap& map);

    /// Return Epetra_FEVector pointer
    boost::shared_ptr<Epetra_FEVector> vec() const;

    /// Assignment operator
    const EpetraVector& operator= (const EpetraVector& x);

  private:

    // Epetra_FEVector pointer
    boost::shared_ptr<Epetra_FEVector> x;

    // Epetra_FEVector pointer
    boost::shared_ptr<Epetra_Vector> x_ghost;

    // Global-to-local map for ghost values
    boost::unordered_map<std::size_t, std::size_t> ghost_global_to_local;

    // Cache of off-process 'set' values (versus 'add') to be communicated
    boost::unordered_map<std::size_t, double> off_process_set_values;

    // Local/global vector
    const std::string type;

  };

}

#endif
#endif
