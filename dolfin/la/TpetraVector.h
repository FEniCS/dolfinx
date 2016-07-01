// Copyright (C) 2014
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

#ifndef __TPETRA_VECTOR_H
#define __TPETRA_VECTOR_H

#ifdef HAS_TRILINOS

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Version.hpp>

#include <dolfin/common/types.h>
#include "GenericVector.h"

namespace dolfin
{

  template<typename T> class Array;

  /// This class provides a simple vector class based on Tpetra.  It
  /// is a wrapper for Teuchos::RCP<Tpetra::MultiVector> implementing
  /// the GenericVector interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the Teuchos::RCP<Tpetra::MultiVector> using the function
  /// vec() and use the standard Tpetra interface.

  class TpetraVector : public GenericVector
  {
  public:

     // Tpetra typedefs with default values
    typedef Tpetra::MultiVector<>::node_type node_type;
    typedef Tpetra::Map<int, dolfin::la_index, node_type> map_type;
    typedef Tpetra::MultiVector<double, int, dolfin::la_index, node_type>
    vector_type;

    /// Create empty vector
    TpetraVector(MPI_Comm comm=MPI_COMM_WORLD);

    /// Create vector of size N
    TpetraVector(MPI_Comm comm, std::size_t N);

    /// Create vector
    //explicit TpetraVector(const SparsityPattern& sparsity_pattern);

    /// Copy constructor
    TpetraVector(const TpetraVector& x);

    /// Create vector wrapper of Tpetra Vec pointer
    //explicit TpetraVector(Teuchos::RCP<vector_type> x);

    /// Destructor
    virtual ~TpetraVector();

    //--- Implementation of the GenericTensor interface ---

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return MPI communicator
    virtual MPI_Comm mpi_comm() const;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericVector interface ---

    /// Return copy of vector
    virtual std::shared_ptr<GenericVector> copy() const;

    /// Initialize vector to global size N
    virtual void init(MPI_Comm comm, std::size_t N);

    /// Initialize vector with given ownership range
    virtual void init(MPI_Comm comm,
                      std::pair<std::size_t, std::size_t> range);

    /// Initialize vector with given ownership range and with ghost
    /// values
    virtual void init(MPI_Comm comm,
                      std::pair<std::size_t, std::size_t> range,
                      const std::vector<std::size_t>& local_to_global_map,
                      const std::vector<la_index>& ghost_indices);

    // Bring init function from GenericVector into scope
    using GenericVector::init;

    /// Return true if vector is empty
    virtual bool empty() const;

    /// Return size of vector
    virtual std::size_t size() const;

    /// Return local size of vector
    virtual std::size_t local_size() const;

    /// Return ownership range of a vector
    virtual std::pair<std::int64_t, std::int64_t> local_range() const;

    /// Determine whether global vector index is owned by this process
    virtual bool owns_index(std::size_t i) const;

    /// Get block of values using global indices (all values must be
    /// owned by local process, ghosts cannot be accessed)
    virtual void get(double* block, std::size_t m,
                     const dolfin::la_index* rows) const;


    /// Get block of values using local indices
    virtual void get_local(double* block, std::size_t m,
                           const dolfin::la_index* rows) const;

    /// Set block of values using global indices
    virtual void set(const double* block, std::size_t m,
                     const dolfin::la_index* rows);

    /// Set block of values using local indices
    virtual void set_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows);

    /// Add block of values using global indices
    virtual void add(const double* block, std::size_t m,
                     const dolfin::la_index* rows);

    /// Add block of values using local indices
    virtual void add_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows);

    /// Get all values on local process
    virtual void get_local(std::vector<double>& values) const;

    /// Set all values on local process
    virtual void set_local(const std::vector<double>& values);

    /// Add values to each entry on local process
    virtual void add_local(const Array<double>& values);

    /// Gather vector entries into a local vector
    virtual void gather(GenericVector& y,
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
    virtual double sum(const Array<std::size_t>& rows) const;

    /// Multiply vector by given number
    virtual const TpetraVector& operator*= (double a);

    /// Multiply vector by another vector pointwise
    virtual const TpetraVector& operator*= (const GenericVector& x);

    /// Divide vector by given number
    virtual const TpetraVector& operator/= (double a);

    /// Add given vector
    virtual const TpetraVector& operator+= (const GenericVector& x);

    /// Add number to all components of a vector
    virtual const TpetraVector& operator+= (double a);

    /// Subtract given vector
    virtual const TpetraVector& operator-= (const GenericVector& x);

    /// Subtract number from all components of a vector
    virtual const TpetraVector& operator-= (double a);

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x);

    /// Assignment operator
    virtual const TpetraVector& operator= (double a);

    // Update ghost values in vector
    virtual void update_ghost_values();

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const;

    //--- Special Tpetra functions ---

    /// Return pointer to Tpetra vector object
    Teuchos::RCP<vector_type> vec() const;

    /// Assignment operator
    const TpetraVector& operator= (const TpetraVector& x);

    /// output map

    static void mapdump(Teuchos::RCP<const map_type> xmap,
                        const std::string desc);

    // Dump x.map and ghost_map
    void mapdump(const std::string desc);

    friend class TpetraMatrix;

  private:

    // Initialise Tpetra vector
    void _init(MPI_Comm comm, std::pair<std::int64_t, std::int64_t> range,
               const std::vector<dolfin::la_index>& local_to_global);

    // Tpetra multivector - actually a view into the ghosted vector,
    // below
    Teuchos::RCP<vector_type> _x;

    // Tpetra multivector with extra rows for ghost values
    Teuchos::RCP<vector_type> _x_ghosted;

  };

}

#endif

#endif
