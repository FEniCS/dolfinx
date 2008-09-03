// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Dag Lindbo, 2008
//
// First added:  2006-07-05
// Last changed: 2008-07-20

#ifndef __UBLAS_MATRIX_H
#define __UBLAS_MATRIX_H

#include <sstream>
#include <iomanip>
#include <boost/tuple/tuple.hpp>

#include <dolfin/log/LogStream.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/Array.h>
#include "LinearAlgebraFactory.h"
#include "SparsityPattern.h"
#include "ublas.h"
#include "uBLASFactory.h"
#include "uBLASVector.h"
#include "GenericMatrix.h"

namespace dolfin
{

  // Forward declarations
  class uBLASVector;
  template< class T> class uBLASFactory;

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

  template<class Mat>
  class uBLASMatrix : public GenericMatrix, public Variable
  {
  public:

    /// Create empty matrix
    uBLASMatrix();
    
    /// Create M x N matrix
    uBLASMatrix(uint M, uint N);

    /// Copy constructor
    explicit uBLASMatrix(const uBLASMatrix& A);

    /// Create matrix from given uBLAS matrix expression
    template <class E>
    explicit uBLASMatrix(const ublas::matrix_expression<E>& A) : Mat(A) {}

    /// Destructor
    virtual ~uBLASMatrix();

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern);

    /// Return copy of tensor
    virtual uBLASMatrix<Mat>* copy() const;

    /// Return size of given dimension
    virtual uint size(uint dim) const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply();

    /// Display tensor
    virtual void disp(uint precision=2) const;

    //--- Implementation of the GenericMatrix interface ---

    /// Initialize M x N matrix
    virtual void init(uint M, uint N);

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const;

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Get non-zero values of given row
    virtual void getrow(uint row, Array<uint>& columns, Array<real>& values) const;

    /// Set values for given row
    virtual void setrow(uint row_idx, const Array<uint>& columns, const Array<real>& values);

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows);

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows);

    /// Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const; 

    /// Multiply matrix by given number
    virtual const uBLASMatrix<Mat>& operator*= (real a);

    /// Divide matrix by given number
    virtual const uBLASMatrix<Mat>& operator/= (real a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A);

    /// Return pointers to underlying compresssed storage data
    virtual boost::tuple<const std::size_t*, const std::size_t*, const double*, int> data() const;

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const
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
    void solveInPlace(uBLASVector& x, const uBLASVector& b);

    /// Compute inverse of matrix
    void invert();

    /// Lump matrix into vector m
    void lump(uBLASVector& m) const;

    /// Compress matrix (eliminate all non-zeros from a sparse matrix) 
    void compress();

    /// Access value of given entry
    real operator() (uint i, uint j) const
    { return A(i, j); }

    /// Assignment operator
    const uBLASMatrix<Mat>& operator= (const uBLASMatrix<Mat>& A);

  private:

    /// General uBLAS LU solver which accepts both vector and matrix right-hand sides
    template<class B>
    void solveInPlace(B& X);

    // uBLAS matrix object
    Mat A;

  };

  //---------------------------------------------------------------------------
  // Implementation of uBLASMatrix
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBLASMatrix<Mat>::uBLASMatrix() : GenericMatrix(), A(0, 0)
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBLASMatrix<Mat>::uBLASMatrix(uint M, uint N) : GenericMatrix(), A(M, N)
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBLASMatrix<Mat>::uBLASMatrix(const uBLASMatrix& A) : GenericMatrix(), A(A.A)
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  uBLASMatrix<Mat>::~uBLASMatrix()
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBLASMatrix< Mat >::init(uint M, uint N)
  {
    // Resize matrix
    if( size(0) != M || size(1) != N )
      A.Mat::resize(M, N, false);  

    // Clear matrix (detroys any structure)
    A.Mat::clear();
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  uBLASMatrix<Mat>* uBLASMatrix<Mat>::copy() const
  {
    return new uBLASMatrix<Mat>(*this);
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uint uBLASMatrix<Mat>::size(uint dim) const
  {
    dolfin_assert( dim < 2 );
    return (dim == 0 ? A.Mat::size1() : A.Mat::size2());  
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBLASMatrix<Mat>::getrow(uint row_idx, Array<uint>& columns, Array<real>& values) const
  {
    dolfin_assert(row_idx < this->size(0));

    // Reference to matrix row (throw away const-ness and trust uBLAS)
    ublas::matrix_row<Mat> row( *(const_cast<Mat*>(&A)) , row_idx);
        
    // Insert values into Arrays
    columns.clear();
    values.clear();
    typename ublas::matrix_row<Mat>::const_iterator component;
    for (component=row.begin(); component != row.end(); ++component) 
    {
      columns.push_back(component.index());
      values.push_back(*component );
    }
  }
  //-----------------------------------------------------------------------------
  template <class Mat>
  void uBLASMatrix<Mat>::setrow(uint row_idx, const Array<uint>& columns, const Array<real>& values)
  {
    dolfin_assert(columns.size() == values.size());
    dolfin_assert(row_idx < this->size(0));

    ublas::matrix_row<Mat> row(A, row_idx);
    dolfin_assert(columns.size() <= row.size());

    row *= 0; 
    for(uint i = 0; i < columns.size(); i++)
      row(columns[i])=values[i];	
  }
  //-----------------------------------------------------------------------------
  template <class Mat>
  void uBLASMatrix<Mat>::set(const real* block, uint m, const uint* rows,
                                                uint n, const uint* cols)
  {
    for (uint i = 0; i < m; i++)
      for (uint j = 0; j < n; j++)
        A(rows[i] , cols[j]) = block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::add(const real* block, uint m, const uint* rows,
                                                uint n, const uint* cols)
  {
    for (uint i = 0; i < m; i++)
      for (uint j = 0; j < n; j++)
        A(rows[i] , cols[j]) += block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBLASMatrix<Mat>::get(real* block, uint m, const uint* rows,
                                          uint n, const uint* cols) const
  {
    for(uint i = 0; i < m; ++i)
      for(uint j = 0; j < n; ++j)
        block[i*n + j] = A(rows[i], cols[j]);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::lump(uBLASVector& m) const
  {
    const uint n = size(1);
    m.init(n);
    ublas::scalar_vector<double> one(n, 1.0);
    ublas::axpy_prod(A, one, m.vec(), true);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::solve(uBLASVector& x, const uBLASVector& b) const
  {
    // Make copy of matrix and vector
    uBLASMatrix<Mat> Atemp;
    Atemp.mat().resize(size(0), size(1));
    Atemp.mat().assign(A);
    x.vec().resize(b.vec().size());
    x.vec().assign(b.vec());

    // Solve
    Atemp.solveInPlace(x.vec());
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::solveInPlace(uBLASVector& x, const uBLASVector& b)
  {
    const uint M = A.size1();
    dolfin_assert(M == b.size());
  
    // Initialise solution vector
    if( x.vec().size() != M )
      x.vec().resize(M);
    x.vec().assign(b.vec());

    // Solve
    solveInPlace(x.vec());
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::invert()
  {
    const uint M = A.size1();
    dolfin_assert(M == A.size2());
  
    // Create indentity matrix
    Mat X(M, M);
    X.assign(ublas::identity_matrix<real>(M));

    // Solve
    solveInPlace(X);
    A.assign_temporary(X);
  }
//-----------------------------------------------------------------------------
  template <class Mat>
  void uBLASMatrix<Mat>::apply()
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::zero()
  {
    // Set all non-zero values to zero without detroying non-zero pattern
    // It might be faster to iterate throught entries?
    A *= 0.0;
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::zero(uint m, const uint* rows) 
  {
    for(uint i = 0; i < m; ++i) 
      ublas::row(A, rows[i]) *= 0.0;  
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::ident(uint m, const uint* rows) 
  {
    const uint n = this->size(1);
    for(uint i = 0; i < m; ++i)
      ublas::row(A, rows[i]) = ublas::unit_vector<double> (n, rows[i]);
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBLASMatrix<Mat>::mult(const GenericVector& x, GenericVector& y, bool transposed) const
  {
    if (transposed == true) 
      error("The transposed version of the uBLAS matrix-vector product is not yet implemented");

    ublas::axpy_prod(A, x.down_cast<uBLASVector>().vec(), y.down_cast<uBLASVector>().vec(), true);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>
  const uBLASMatrix<Mat>& uBLASMatrix<Mat>::operator*= (real a)
   { 
    A *= a; 
    return *this; 
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  const uBLASMatrix<Mat>& uBLASMatrix<Mat>::operator/= (real a)
  { 
    A /= a; 
    return *this; 
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  const GenericMatrix& uBLASMatrix<Mat>::operator= (const GenericMatrix& A)
  { 
    this->A = A.down_cast< uBLASMatrix<Mat> >().mat(); 
    return *this;
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  inline const uBLASMatrix<Mat>& uBLASMatrix<Mat>::operator= (const uBLASMatrix<Mat>& A)
  {
    this->A = A.mat(); 
    return *this;
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  inline void uBLASMatrix<Mat>::compress()
  {
    Mat A_temp(this->size(0), this->size(1));
    A_temp.assign(A);
    A.swap(A_temp);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBLASMatrix<Mat>::disp(uint precision) const
  {
    typename Mat::const_iterator1 it1;  // Iterator over rows
    typename Mat::const_iterator2 it2;  // Iterator over entries

    for (it1 = A.begin1(); it1 != A.end1(); ++it1)
    {    
      dolfin::cout << "|";
      for (it2 = it1.begin(); it2 != it1.end(); ++it2)
      {
        std::stringstream entry;
        entry << std::setiosflags(std::ios::scientific);
        entry << std::setprecision(precision);
        entry << " (" << it2.index1() << ", " << it2.index2() << ", " << *it2 << ")";
        dolfin::cout << entry.str().c_str();
      }
      dolfin::cout  << " |" << dolfin::endl;
    }  
  }
  //-----------------------------------------------------------------------------
  // Specialised member functions (must be inlined to avoid link errors)
  //-----------------------------------------------------------------------------
  template <> 
  inline void uBLASMatrix<ublas_sparse_matrix>::init(const GenericSparsityPattern& sparsity_pattern)
  {
    init(sparsity_pattern.size(0), sparsity_pattern.size(1));

    // Reserve space for non-zeroes
    A.reserve(sparsity_pattern.numNonZero());

    const SparsityPattern* pattern_pointer = dynamic_cast<const SparsityPattern*>(&sparsity_pattern);
    if (not pattern_pointer)
      error("Cannot convert GenericSparsityPattern to concrete SparsityPattern type. Aborting.");

    const std::vector< std::vector<uint> >& pattern = pattern_pointer->pattern();

    // Sort sparsity pattern
    pattern_pointer->sort();

    std::vector< std::vector<uint> >::const_iterator row;
    std::vector<uint>::const_iterator element;
    for(row = pattern.begin(); row != pattern.end(); ++row)
      for(element = row->begin(); element != row->end(); ++element)
        A.push_back(row - pattern.begin(), *element, 0.0);
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  inline void uBLASMatrix<Mat>::init(const GenericSparsityPattern& sparsity_pattern)
  {
    init(sparsity_pattern.size(0), sparsity_pattern.size(1));
  }
  //---------------------------------------------------------------------------
  template <>
  inline boost::tuple<const std::size_t*, const std::size_t*, const double*, int> 
                                  uBLASMatrix<ublas_sparse_matrix>::data() const
  { 
    // Make sure matrix assembly is complete
    const_cast< ublas_sparse_matrix& >(A).complete_index1_data(); 

    typedef boost::tuple<const std::size_t*, const std::size_t*, const double*, int> tuple;
    return tuple(&A.index1_data()[0], &A.index2_data()[0], &A.value_data()[0], A.nnz());
  } 
  //---------------------------------------------------------------------------
  template <class Mat>
  inline boost::tuple<const std::size_t*, const std::size_t*, const double*, int> 
                                                    uBLASMatrix<Mat>::data() const
  { 
    error("Unable to return pointers to underlying data for this uBLASMatrix type."); 
    return boost::tuple<const std::size_t*, const std::size_t*, const double*, int>(0, 0, 0, 0);
  } 
  //---------------------------------------------------------------------------
  template<class Mat> template<class B>
  void uBLASMatrix<Mat>::solveInPlace(B& X)
  {
    const uint M = A.size1();
    dolfin_assert( M == A.size2() );
  
    // Create permutation matrix
    ublas::permutation_matrix<std::size_t> pmatrix(M);

    // Factorise (with pivoting)
    uint singular = ublas::lu_factorize(A, pmatrix);
    if( singular > 0)
      error("Singularity detected in uBLAS matrix factorization on line %u.", singular-1); 

    // Back substitute 
    ublas::lu_substitute(A, pmatrix, X);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  inline LogStream& operator<< (LogStream& stream, const uBLASMatrix<Mat>& B)
  {
    // Check if matrix has been defined
    if ( B.size(0) == 0 || B.size(1) == 0 )
    {
      stream << "[ uBLASMatrix matrix (empty) ]";
      return stream;
    }

    uint M = B.size(0);
    uint N = B.size(1);
    stream << "[ uBLASMatrix matrix of size " << M << " x " << N << " ]";

    return stream;
  }
  //-----------------------------------------------------------------------------
}

#endif
