// Copyright (C) 2006-2009 Garth N. Wells
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
// Modified by Anders Logg 2006-2012
// Modified by Ola Skavhaug 2007-2008
// Modified by Kent-Andre Mardal 2008
// Modified by Martin Sandve Alnes 2008
// Modified by Dag Lindbo 2008
//
// First added:  2006-07-05
// Last changed: 2012-08-20

#ifndef __UBLAS_MATRIX_H
#define __UBLAS_MATRIX_H

#include <sstream>
#include <iomanip>
#include <boost/tuple/tuple.hpp>

#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"
#include "SparsityPattern.h"
#include "TensorLayout.h"
#include "ublas.h"
#include "uBLASFactory.h"
#include "uBLASVector.h"

namespace dolfin
{

  // Forward declarations
  class uBLASVector;
  template<typename T> class uBLASFactory;

  namespace ublas = boost::numeric::ublas;

  /// This class provides a simple matrix class based on uBLAS.
  /// It is a simple wrapper for a uBLAS matrix implementing the
  /// GenericMatrix interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the underlying uBLAS matrix and use the standard
  /// uBLAS interface which is documented at
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.
  ///
  /// Developer note: specialised member functions must be
  /// inlined to avoid link errors.

  template<typename Mat>
  class uBLASMatrix : public GenericMatrix
  {
  public:

    /// Create empty matrix
    uBLASMatrix();

    /// Create M x N matrix
    uBLASMatrix(std::size_t M, std::size_t N);

    /// Copy constructor
    uBLASMatrix(const uBLASMatrix& A);

    /// Create matrix from given uBLAS matrix expression
    template <typename E>
    explicit uBLASMatrix(const ublas::matrix_expression<E>& A) : Mat(A) {}

    /// Destructor
    virtual ~uBLASMatrix();

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using tenor layout
    virtual void init(const TensorLayout& tensor_layout);

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const;

    /// Return local ownership range
    virtual std::pair<std::size_t, std::size_t> local_range(std::size_t dim) const
    { return std::make_pair(0, size(dim)); }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericMatrix interface ---

    /// Return copy of matrix
    virtual boost::shared_ptr<GenericMatrix> copy() const;

    /// Resize matrix to M x N
    virtual void resize(std::size_t M, std::size_t N);

    /// Resize vector z to be compatible with the matrix-vector product
    /// y = Ax. In the parallel case, both size and layout are
    /// important.
    ///
    /// *Arguments*
    ///     dim (std::size_t)
    ///         The dimension (axis): dim = 0 --> z = y, dim = 1 --> z = x
    virtual void resize(GenericVector& z, std::size_t dim) const;

    /// Get block of values
    virtual void get(double* block, std::size_t m, const DolfinIndex* rows, std::size_t n, const DolfinIndex* cols) const;

    /// Set block of values
    virtual void set(const double* block, std::size_t m, const DolfinIndex* rows, std::size_t n, const DolfinIndex* cols);

    /// Add block of values
    virtual void add(const double* block, std::size_t m, const DolfinIndex* rows, std::size_t n, const DolfinIndex* cols);

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A,
                      bool same_nonzero_pattern);

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const;

    /// Get non-zero values of given row
    virtual void getrow(std::size_t row, std::vector<std::size_t>& columns,
                        std::vector<double>& values) const;

    /// Set values for given row
    virtual void setrow(std::size_t row_idx, const std::vector<std::size_t>& columns,
                        const std::vector<double>& values);

    /// Set given rows to zero
    virtual void zero(std::size_t m, const DolfinIndex* rows);

    /// Set given rows to identity matrix
    virtual void ident(std::size_t m, const DolfinIndex* rows);

    /// Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const;

    /// Matrix-vector product, y = A^T x
    virtual void transpmult(const GenericVector& x, GenericVector& y) const;

    /// Multiply matrix by given number
    virtual const uBLASMatrix<Mat>& operator*= (double a);

    /// Divide matrix by given number
    virtual const uBLASMatrix<Mat>& operator/= (double a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A);

    /// Return pointers to underlying compresssed storage data
    /// See GenericMatrix for documentation.
    virtual boost::tuples::tuple<const std::size_t*, const std::size_t*, const double*, int> data() const;

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const
    { return uBLASFactory<Mat>::instance(); }

    //--- Special uBLAS functions ---

    /// Return reference to uBLAS matrix (const version)
    const Mat& mat() const
    { return A; }

    /// Return reference to uBLAS matrix (non-const version)
    Mat& mat()
    { return A; }

    /// Solve Ax = b out-of-place using uBLAS (A is not destroyed)
    void solve(uBLASVector& x, const uBLASVector& b) const;

    /// Solve Ax = b in-place using uBLAS(A is destroyed)
    void solve_in_place(uBLASVector& x, const uBLASVector& b);

    /// Compute inverse of matrix
    void invert();

    /// Lump matrix into vector m
    void lump(uBLASVector& m) const;

    /// Compress matrix (eliminate all non-zeros from a sparse matrix)
    void compress();

    /// Access value of given entry
    double operator() (DolfinIndex i, DolfinIndex j) const
    { return A(i, j); }

    /// Assignment operator
    const uBLASMatrix<Mat>& operator= (const uBLASMatrix<Mat>& A);

  private:

    /// General uBLAS LU solver which accepts both vector and matrix right-hand sides
    template<typename B>
    void solve_in_place(B& X);

    // uBLAS matrix object
    Mat A;

  };

  //---------------------------------------------------------------------------
  // Implementation of uBLASMatrix
  //---------------------------------------------------------------------------
  template <typename Mat>
  uBLASMatrix<Mat>::uBLASMatrix() : GenericMatrix(), A(0, 0)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  uBLASMatrix<Mat>::uBLASMatrix(std::size_t M, std::size_t N) : GenericMatrix(), A(M, N)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  uBLASMatrix<Mat>::uBLASMatrix(const uBLASMatrix& A) : GenericMatrix(), A(A.A)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  uBLASMatrix<Mat>::~uBLASMatrix()
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  boost::shared_ptr<GenericMatrix> uBLASMatrix<Mat>::copy() const
  {
    boost::shared_ptr<GenericMatrix> A(new uBLASMatrix<Mat>(*this));
    return A;
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix< Mat >::resize(std::size_t M, std::size_t N)
  {
    // Resize matrix
    if( size(0) != M || size(1) != N )
      A.Mat::resize(M, N, false);
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  std::size_t uBLASMatrix<Mat>::size(std::size_t dim) const
  {
    if (dim > 1)
    {
      dolfin_error("uBLASMatrix.cpp",
                   "access size of uBLAS matrix",
                   "Illegal axis (%d), must be 0 or 1", dim);
    }

    dolfin_assert(dim < 2);
    return (dim == 0 ? A.Mat::size1() : A.Mat::size2());
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  double uBLASMatrix<Mat>::norm(std::string norm_type) const
  {
    if (norm_type == "l1")
      return norm_1(A);
    else if (norm_type == "linf")
      return norm_inf(A);
    else if (norm_type == "frobenius")
      return norm_frobenius(A);
    else
    {
      dolfin_error("uBLASMatrix.h",
                   "compute norm of uBLAS matrix",
                   "Unknown norm type (\"%s\")",
                   norm_type.c_str());
      return 0.0;
    }
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::getrow(std::size_t row_idx,
                                std::vector<std::size_t>& columns,
                                std::vector<double>& values) const
  {
    dolfin_assert(row_idx < this->size(0));

    // Reference to matrix row
    const ublas::matrix_row<const Mat> row(A, row_idx);

    // Insert values into std::vectors
    columns.clear();
    values.clear();
    typename ublas::matrix_row<const Mat>::const_iterator component;
    for (component = row.begin(); component != row.end(); ++component)
    {
      columns.push_back(component.index());
      values.push_back(*component );
    }
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::setrow(std::size_t row_idx,
                                const std::vector<std::size_t>& columns,
                                const std::vector<double>& values)
  {
    dolfin_assert(columns.size() == values.size());
    dolfin_assert(row_idx < this->size(0));

    ublas::matrix_row<Mat> row(A, row_idx);
    dolfin_assert(columns.size() <= row.size());

    row *= 0;
    for(std::size_t i = 0; i < columns.size(); i++)
      row(columns[i])=values[i];
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::resize(GenericVector& z, std::size_t dim) const
  {
    z.resize(size(dim));
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::set(const double* block, std::size_t m, const DolfinIndex* rows,
                             std::size_t n, const DolfinIndex* cols)
  {
    for (std::size_t i = 0; i < m; i++)
      for (std::size_t j = 0; j < n; j++)
        A(rows[i] , cols[j]) = block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::add(const double* block, std::size_t m, const DolfinIndex* rows,
                             std::size_t n, const DolfinIndex* cols)
  {
    for (std::size_t i = 0; i < m; i++)
      for (std::size_t j = 0; j < n; j++)
        A(rows[i] , cols[j]) += block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::get(double* block, std::size_t m, const DolfinIndex* rows,
                             std::size_t n, const DolfinIndex* cols) const
  {
    for(std::size_t i = 0; i < m; ++i)
      for(std::size_t j = 0; j < n; ++j)
        block[i*n + j] = A(rows[i], cols[j]);
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::lump(uBLASVector& m) const
  {
    const std::size_t n = size(1);
    m.resize(n);
    m.zero();
    ublas::scalar_vector<double> one(n, 1.0);
    ublas::axpy_prod(A, one, m.vec(), true);
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::solve(uBLASVector& x, const uBLASVector& b) const
  {
    // Make copy of matrix and vector
    uBLASMatrix<Mat> Atemp;
    Atemp.mat().resize(size(0), size(1));
    Atemp.mat().assign(A);
    x.vec().resize(b.vec().size());
    x.vec().assign(b.vec());

    // Solve
    Atemp.solve_in_place(x.vec());
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::solve_in_place(uBLASVector& x, const uBLASVector& b)
  {
    const std::size_t M = A.size1();
    dolfin_assert(M == b.size());

    // Initialise solution vector
    if( x.vec().size() != M )
      x.vec().resize(M);
    x.vec().assign(b.vec());

    // Solve
    solve_in_place(x.vec());
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::invert()
  {
    const std::size_t M = A.size1();
    dolfin_assert(M == A.size2());

    // Create indentity matrix
    Mat X(M, M);
    X.assign(ublas::identity_matrix<double>(M));

    // Solve
    solve_in_place(X);
    A.assign_temporary(X);
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::zero()
  {
    // Iterate through no-zero pattern and zero entries
    typename Mat::iterator1 row;    // Iterator over rows
    typename Mat::iterator2 entry;  // Iterator over entries
    for (row = A.begin1(); row != A.end1(); ++row)
      for (entry = row.begin(); entry != row.end(); ++entry)
        *entry = 0;
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::zero(std::size_t m, const DolfinIndex* rows)
  {
    for(std::size_t i = 0; i < m; ++i)
      ublas::row(A, rows[i]) *= 0.0;
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::ident(std::size_t m, const DolfinIndex* rows)
  {
    // Copy row indices to a vector
    std::vector<std::size_t> _rows(rows, rows + m);

    std::size_t counter = 0;
    typename Mat::iterator1 row;    // Iterator over rows
    typename Mat::iterator2 entry;  // Iterator over entries
    for (row = A.begin1(); row != A.end1(); ++row)
    {
      entry = row.begin();
      if (std::find(_rows.begin(), _rows.end(), entry.index1()) != _rows.end())
      {
        // Iterate over entries to zero and place one on the diagonal
        for (entry = row.begin(); entry != row.end(); ++entry)
        {
          if (entry.index1() == entry.index2())
            *entry = 1.0;
          else
            *entry = 0.0;
        }
        ++ counter;
      }
      if (counter == _rows.size())
        continue;
    }
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::mult(const GenericVector& x, GenericVector& y) const
  {
    const uBLASVector& xx = as_type<const uBLASVector>(x);
    uBLASVector& yy = as_type<uBLASVector>(y);

    if (size(1) != xx.size())
    {
      dolfin_error("uBLASMatrix.cpp",
                   "compute matrix-vector product with uBLAS matrix",
                   "Non-matching dimensions for matrix-vector product");
    }

    // Resize RHS if empty
    if (yy.size() == 0)
    {
      resize(yy, 0);
    }

    if (size(0) != yy.size())
    {
      dolfin_error("uBLASMatrix.cpp",
                   "compute matrix-vector product with uBLAS matrix",
                   "Vector for matrix-vector result has wrong size");
    }

    ublas::axpy_prod(A, xx.vec(), yy.vec(), true);
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  void uBLASMatrix<Mat>::transpmult(const GenericVector& x, GenericVector& y) const
  {
    dolfin_error("uBLASMatrix.h",
                 "compute transpose matrix-vector product",
                 "Not supported by the uBLAS linear algebra backend");
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  const uBLASMatrix<Mat>& uBLASMatrix<Mat>::operator*= (double a)
   {
    A *= a;
    return *this;
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  const uBLASMatrix<Mat>& uBLASMatrix<Mat>::operator/= (double a)
  {
    A /= a;
    return *this;
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  const GenericMatrix& uBLASMatrix<Mat>::operator= (const GenericMatrix& A)
  {
    *this = as_type<const uBLASMatrix<Mat> >(A);
    return *this;
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  inline const uBLASMatrix<Mat>& uBLASMatrix<Mat>::operator= (const uBLASMatrix<Mat>& A)
  {
    // Check for self-assignment
    if (this != &A)
    {
      // Assume uBLAS take care of deleting an existing Matrix
      // using its assignment operator
      this->A = A.mat();
    }
    return *this;
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  inline void uBLASMatrix<Mat>::compress()
  {
    Mat A_temp(this->size(0), this->size(1));
    A_temp.assign(A);
    A.swap(A_temp);
  }
  //-----------------------------------------------------------------------------
  template <typename Mat>
  std::string uBLASMatrix<Mat>::str(bool verbose) const
  {
    typename Mat::const_iterator1 it1;  // Iterator over rows
    typename Mat::const_iterator2 it2;  // Iterator over entries

    std::stringstream s;

    if (verbose)
    {
      s << str(false) << std::endl << std::endl;
      for (it1 = A.begin1(); it1 != A.end1(); ++it1)
      {
        s << "|";
        for (it2 = it1.begin(); it2 != it1.end(); ++it2)
        {
          std::stringstream entry;
          entry << std::setiosflags(std::ios::scientific);
          entry << std::setprecision(16);
          entry << " (" << it2.index1() << ", " << it2.index2() << ", " << *it2 << ")";
          s << entry.str();
        }
        s << " |" << std::endl;
      }
    }
    else
    {
      s << "<uBLASMatrix of size " << size(0) << " x " << size(1) << ">";
    }

    return s.str();
  }
  //-----------------------------------------------------------------------------
  // Specialised member functions (must be inlined to avoid link errors)
  //-----------------------------------------------------------------------------
  template <>
  inline void uBLASMatrix<ublas_sparse_matrix>::init(const TensorLayout& tensor_layout)
  {
    resize(tensor_layout.size(0), tensor_layout.size(1));
    A.clear();

    // Get sparsity pattern
    dolfin_assert(tensor_layout.sparsity_pattern());
    const SparsityPattern* pattern_pointer = dynamic_cast<const SparsityPattern*>(tensor_layout.sparsity_pattern().get());
    if (!pattern_pointer)
    {
      dolfin_error("uBLASMatrix.h",
                   "initialize uBLAS matrix",
                   "Cannot convert GenericSparsityPattern to concrete SparsityPattern type");
    }

    // Reserve space for non-zeroes and get non-zero pattern
    A.reserve(pattern_pointer->num_nonzeros());
    const std::vector<std::vector<std::size_t> > pattern = pattern_pointer->diagonal_pattern(SparsityPattern::sorted);

    // Add entries
    std::vector<std::vector<std::size_t> >::const_iterator row;
    Set<std::size_t>::const_iterator element;
    for(row = pattern.begin(); row != pattern.end(); ++row)
      for(element = row->begin(); element != row->end(); ++element)
        A.push_back(row - pattern.begin(), *element, 0.0);
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  inline void uBLASMatrix<Mat>::init(const TensorLayout& tensor_layout)
  {
    resize(tensor_layout.size(0), tensor_layout.size(1));
    A.clear();
  }
  //---------------------------------------------------------------------------
  template <>
  inline void uBLASMatrix<ublas_sparse_matrix>::apply(std::string mode)
  {
    Timer timer("Apply (matrix)");

    // Make sure matrix assembly is complete
    A.complete_index1_data();
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  inline void uBLASMatrix<Mat>::apply(std::string mode)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  inline void uBLASMatrix<Mat>::axpy(double a, const GenericMatrix& A,
                                     bool same_nonzero_pattern)
  {
    // Check for same size
    if (size(0) != A.size(0) or size(1) != A.size(1))
    {
      dolfin_error("uBLASMatrix.h",
                   "perform axpy operation with uBLAS matrix",
                   "Dimensions don't match");
    }

    this->A += (a)*(as_type<const uBLASMatrix>(A).mat());
  }
  //---------------------------------------------------------------------------
  template <>
  inline boost::tuples::tuple<const std::size_t*, const std::size_t*,
                         const double*, int> uBLASMatrix<ublas_sparse_matrix>::data() const
  {
    typedef boost::tuples::tuple<const std::size_t*, const std::size_t*, const double*, int> tuple;
    return tuple(&A.index1_data()[0], &A.index2_data()[0], &A.value_data()[0], A.nnz());
  }
  //---------------------------------------------------------------------------
  template <typename Mat>
  inline boost::tuples::tuple<const std::size_t*, const std::size_t*, const double*, int>
  uBLASMatrix<Mat>::data() const
  {
    dolfin_error("GenericMatrix.h",
                 "return pointers to underlying matrix data",
                 "Not implemented for this uBLAS matrix type");
    return boost::tuples::tuple<const std::size_t*,
                                const std::size_t*,
                                const double*,
                                int>(0, 0, 0, 0);
  }
  //---------------------------------------------------------------------------
  template<typename Mat> template<typename B>
  void uBLASMatrix<Mat>::solve_in_place(B& X)
  {
    const std::size_t M = A.size1();
    dolfin_assert(M == A.size2());

    // Create permutation matrix
    ublas::permutation_matrix<std::size_t> pmatrix(M);

    // Factorise (with pivoting)
    std::size_t singular = ublas::lu_factorize(A, pmatrix);
    if (singular > 0)
    {
      dolfin_error("uBLASMatrix.h",
                   "solve in-place using uBLAS matrix",
                   "Singularity detected in matrix factorization on row %u",
                   singular - 1);
    }

    // Back substitute
    ublas::lu_substitute(A, pmatrix, X);
  }
  //-----------------------------------------------------------------------------
}

#endif
