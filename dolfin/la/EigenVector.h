// Copyright (C) 2015 Garth N. Wells
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

#ifndef __EIGEN_VECTOR_H
#define __EIGEN_VECTOR_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <dolfin/common/types.h>
#include <Eigen/Dense>

#include <dolfin/common/MPI.h>
#include "GenericVector.h"

namespace dolfin
{

  template<typename T> class Array;

  /// This class provides a simple vector class based on Eigen.
  /// It is a simple wrapper for a Eigen vector implementing the
  /// GenericVector interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the underlying Eigen vector and use the standard Eigen
  /// interface which is documented at http://eigen.tuxfamily.org

  class EigenVector : public GenericVector
  {
  public:

    /// Create empty vector (on MPI_COMM_SELF)
    EigenVector();

    /// Create empty vector
    explicit EigenVector(MPI_Comm comm);

    /// Create vector of size N
    EigenVector(MPI_Comm comm, std::size_t N);

    /// Copy constructor
    EigenVector(const EigenVector& x);

    /// Construct vector from an Eigen shared_ptr
    explicit EigenVector(std::shared_ptr<Eigen::VectorXd> x);

    /// Destructor
    virtual ~EigenVector();

    //--- Implementation of the GenericTensor interface ---

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return MPI communicator
    virtual MPI_Comm mpi_comm() const
    { return _mpi_comm; }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericVector interface ---

    /// Create copy of tensor
    virtual std::shared_ptr<GenericVector> copy() const;

    /// Initialize vector to size N
    virtual void init(std::size_t N)
    {
      if (!empty())
      {
        dolfin_error("EigenVector.cpp",
                     "calling EigenVector::init(...)",
                     "Cannot call init for a non-empty vector. Use EigenVector::resize instead");
      }
      resize(N);
    }

    /// Resize vector with given ownership range
    virtual void init(std::pair<std::size_t, std::size_t> range)
    {
      if (!empty())
      {
        dolfin_error("EigenVector.cpp",
                     "calling EigenVector::init(...)",
                     "Cannot call init for a non-empty vector. Use EigenVector::resize instead");
      }

      dolfin_assert(range.first == 0);
      const std::size_t size = range.second - range.first;
      resize(size);
    }

    /// Resize vector with given ownership range and with ghost values
    virtual void init(std::pair<std::size_t, std::size_t> range,
                      const std::vector<std::size_t>& local_to_global_map,
                      const std::vector<la_index>& ghost_indices)
    {
      if (!empty())
      {
        dolfin_error("EigenVector.cpp",
                     "calling EigenVector::init(...)",
                     "Cannot call init for a non-empty vector. Use EigenVector::resize instead");
      }

      if (!ghost_indices.empty())
      {
        dolfin_error("EigenVector.cpp",
                     "calling EigenVector::init(...)",
                     "EigenVector does not support ghost values");
      }

      dolfin_assert(range.first == 0);
      const std::size_t size = range.second - range.first;
      resize(size);
    }

    // Bring init function from GenericVector into scope
    using GenericVector::init;

    /// Return true if vector is empty
    virtual bool empty() const;

    /// Return true if vector is empty
    virtual std::size_t size() const;

    /// Return local size of vector
    virtual std::size_t local_size() const
    { return size(); }

    /// Return local ownership range of a vector
    virtual std::pair<std::int64_t, std::int64_t> local_range() const;

    /// Determine whether global vector index is owned by this process
    virtual bool owns_index(std::size_t i) const;

    /// Get block of values using global indices
    virtual void get(double* block, std::size_t m,
                     const dolfin::la_index* rows) const
    { get_local(block, m, rows); }

    /// Get block of values using local indices
    virtual void get_local(double* block, std::size_t m,
                           const dolfin::la_index* rows) const;

    /// Set block of values using global indices
    virtual void set(const double* block, std::size_t m,
                     const dolfin::la_index* rows);

    /// Set block of values using local indices
    virtual void set_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows)
    { set(block, m, rows); }

    /// Add block of values using global indices
    virtual void add(const double* block, std::size_t m,
                     const dolfin::la_index* rows);

    /// Add block of values using local indices
    virtual void add_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows)
    { add(block, m, rows); }

    /// Get all values on local process
    virtual void get_local(std::vector<double>& values) const;

    /// Set all values on local process
    virtual void set_local(const std::vector<double>& values);

    /// Add values to each entry on local process
    virtual void add_local(const Array<double>& values);

    /// Gather entries into local vector x
    virtual void gather(GenericVector& x,
                        const std::vector<dolfin::la_index>& indices) const;

    /// Gather entries into x
    virtual void gather(std::vector<double>& x,
                        const std::vector<dolfin::la_index>& indices) const;

    /// Gather all entries into x on process 0
    virtual void gather_on_zero(std::vector<double>& x) const;

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x);

    /// Replace all entries in the vector by their absolute values
    virtual void abs();

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

    /// Return sum of selected rows in vector. Repeated entries are
    /// only summed once.
    virtual double sum(const Array<std::size_t>& rows) const;

    /// Multiply vector by given number
    virtual const EigenVector& operator*= (double a);

    /// Multiply vector by another vector pointwise
    virtual const EigenVector& operator*= (const GenericVector& x);

    /// Divide vector by given number
    virtual const EigenVector& operator/= (double a);

    /// Add given vector
    virtual const EigenVector& operator+= (const GenericVector& x);

    /// Add number to all components of a vector
    virtual const EigenVector& operator+= (double a);

    /// Subtract given vector
    virtual const EigenVector& operator-= (const GenericVector& x);

    /// Subtract number from all components of a vector
    virtual const EigenVector& operator-= (double a);

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x);

    /// Assignment operator
    virtual const EigenVector& operator= (double a);

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const;

    //--- Special Eigen functions ---

    /// Resize vector to size N
    virtual void resize(std::size_t N);

    /// Return reference to Eigen vector (const version)
    const Eigen::VectorXd& vec() const
    { return *_x; }

    /// Return reference to Eigen vector (non-const version)
    Eigen::VectorXd& vec()
    { return *_x; }

    /// Access value of given entry (const version)
    virtual double operator[] (dolfin::la_index i) const
    { return (*_x)(i); }

    /// Access value of given entry (non-const version)
    double& operator[] (dolfin::la_index i)
    { return (*_x)(i); }

    /// Assignment operator
    const EigenVector& operator= (const EigenVector& x);

    /// Return pointer to underlying data
    double* data();

    /// Return pointer to underlying data (const version)
    const double* data() const;

  private:

    static void check_mpi_size(const MPI_Comm comm)
    {
      if (dolfin::MPI::size(comm) > 1)
      {
        dolfin_error("EigenVector.cpp",
                     "creating EigenVector",
                     "EigenVector does not support parallel communicators");
      }
    }

    // Pointer to Eigen vector object
    std::shared_ptr<Eigen::VectorXd> _x;

    // MPI communicator
    MPI_Comm _mpi_comm;

  };

}

#endif
