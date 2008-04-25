// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
//
// First added:  2006-07-05
// Last changed: 2008-04-24

#ifndef __UBLAS_MATRIX_H
#define __UBLAS_MATRIX_H

#include <sstream>
#include <iomanip>

#include <dolfin/log/LogStream.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/Array.h>
#include "LinearAlgebraFactory.h"
#include "SparsityPattern.h"
#include "ublas.h"
#include "uBlasVector.h"
#include "uBlasLUSolver.h"
#include "GenericMatrix.h"

namespace dolfin
{

  class uBlasVector;
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
  class uBlasMatrix : public GenericMatrix, public Variable
  {
  public:

    /// Create empty matrix
    explicit uBlasMatrix();
    
    /// Create M x N matrix
    explicit uBlasMatrix(uint M, uint N);

    /// Copy constructor
    explicit uBlasMatrix(const uBlasMatrix& A);

    /// Create matrix from given uBLAS matrix expression
    template <class E>
    explicit uBlasMatrix(const ublas::matrix_expression<E>& A) : Mat(A) {}

    /// Destructor
    ~uBlasMatrix();

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    void init(const GenericSparsityPattern& sparsity_pattern);

    /// Return copy of tensor
    uBlasMatrix<Mat>* copy() const;

    /// Return size of given dimension
    uint size(uint dim) const;

    /// Set all entries to zero and keep any sparse structure
    void zero();

    /// Finalize assembly of tensor
    void apply();

    /// Display tensor
    void disp(uint precision=2) const;

    //--- Implementation of the GenericMatrix interface ---

    /// Initialize M x N matrix
    void init(uint M, uint N);

    /// Get block of values
    void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const;

    /// Set block of values
    void set(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add block of values
    void add(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Get non-zero values of given row
    void getrow(uint row, Array<uint>& columns, Array<real>& values) const;

    /// Set given rows to zero
    void zero(uint m, const uint* rows);

    /// Set given rows to identity matrix
    void ident(uint m, const uint* rows);

    /// Matrix-vector product, y = Ax
    void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const; 

    /// Multiply matrix by given number
    const uBlasMatrix<Mat>& operator*= (real a)
    { A *= a; return *this; }

    /// Assignment operator
    const GenericMatrix& operator= (const GenericMatrix& A)
    { this->A = A.down_cast< uBlasMatrix<Mat> >().mat(); return *this; }

    /// Assignment operator
    const uBlasMatrix<Mat>& operator= (const uBlasMatrix<Mat>& A)
    { this->A = A.mat(); return *this; }

    //--- Convenience functions ---

    /// Divide matrix by given number
    const uBlasMatrix<Mat>& operator/= (real a)
    { A /= a; return *this; }

    //--- Special functions ---

    /// Return linear algebra backend factory
    LinearAlgebraFactory& factory() const;
    
    //--- Special uBLAS functions ---

    /// Return reference to uBLAS matrix (const version)
    const Mat& mat() const
    { return A; }

    /// Return reference to uBLAS matrix (non-const version)
    Mat& mat()
    { return A; }

    /// Solve Ax = b out-of-place (A is not destroyed)
    void solve(uBlasVector& x, const uBlasVector& b) const;

    /// Compute inverse of matrix
    void invert();

    /// Lump matrix into vector m
    void lump(uBlasVector& m) const;

    /// Compress matrix (eliminate all non-zeros from a sparse matrix) 
    void compress();

    /// Access value of given entry
    real operator() (uint i, uint j) const
    { return A(i, j); }

  private:

    // uBLAS matrix object
    Mat A;

  };

  //---------------------------------------------------------------------------
  // Implementation of uBlasMatrix
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>::uBlasMatrix() : GenericMatrix(), A(0, 0)
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>::uBlasMatrix(uint M, uint N) : GenericMatrix(), A(M, N)
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>::uBlasMatrix(const uBlasMatrix& A) : GenericMatrix(), A(A.A)
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  uBlasMatrix<Mat>::~uBlasMatrix()
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix< Mat >::init(uint M, uint N)
  {
    // Resize matrix
    if( size(0) != M || size(1) != N )
      A.Mat::resize(M, N, false);  

    // Clear matrix (detroys any structure)
    A.Mat::clear();
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  uBlasMatrix<Mat>* uBlasMatrix<Mat>::copy() const
  {
    return new uBlasMatrix<Mat>(*this);
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uint uBlasMatrix<Mat>::size(uint dim) const
  {
    dolfin_assert( dim < 2 );
    return (dim == 0 ? A.Mat::size1() : A.Mat::size2());  
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix< Mat >::getrow(uint row, Array<uint>& columns, Array<real>& values) const
  {
    // Reference to matrix row (throw away const-ness and trust uBlas)
    ublas::matrix_row< Mat > r( *(const_cast<Mat*>(&A)) , row);

    typename ublas::matrix_row< Mat >::const_iterator component;

    // Insert values into Arrays
    columns.clear();
    values.clear();
    for (component=r.begin(); component != r.end(); ++component) 
    {
      columns.push_back(component.index());
      values.push_back(*component );
    }
  }
  //-----------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix<Mat>::set(const real* block,
                                    uint m, const uint* rows,
                                    uint n, const uint* cols)
  {
    for (uint i = 0; i < m; i++)
      for (uint j = 0; j < n; j++)
        A(rows[i] , cols[j]) = block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::add(const real* block,
                                    uint m, const uint* rows,
                                    uint n, const uint* cols)
  {
    for (uint i = 0; i < m; i++)
      for (uint j = 0; j < n; j++)
        A(rows[i] , cols[j]) += block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix<Mat>::get(real* block,
                                    uint m, const uint* rows,
                                    uint n, const uint* cols) const
  {
    for(uint i = 0; i < m; ++i)
      for(uint j = 0; j < n; ++j)
        block[i*n + j] = A(rows[i], cols[j]);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::lump(uBlasVector& m) const
  {
    const uint n = size(1);
    m.init(n);
    ublas::scalar_vector<double> one(n, 1.0);
    ublas::axpy_prod(A, one, m.vec(), true);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::solve(uBlasVector& x, const uBlasVector& b) const
  {    
    uBlasLUSolver solver;
    solver.solve(*this, x, b);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::invert()
  {
    uBlasLUSolver solver;
    solver.invert(A);
  }
//-----------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix<Mat>::apply()
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::zero()
  {
    // Set all non-zero values to zero without detroying non-zero pattern
    // It might be faster to iterate throught entries?
    A *= 0.0;
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::zero(uint m, const uint* rows) 
  {
    for(uint i = 0; i < m; ++i) {
      ublas::row(A, rows[i]) *= 0.0;  
    }
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::ident(uint m, const uint* rows) 
  {
    const uint n = this->size(1);
    for(uint i = 0; i < m; ++i)
      ublas::row(A, rows[i]) = ublas::unit_vector<double> (n, rows[i]);
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix<Mat>::mult(const GenericVector& x, GenericVector& y, bool transposed) const
  {
    const uBlasVector& xx = x.down_cast<uBlasVector>();
    uBlasVector& yy = y.down_cast<uBlasVector>();

    if (transposed == true) error("The transposed version of the uBLAS matrix-vector product is not yet implemented");
    
    ublas::axpy_prod(A, xx.vec(), yy.vec(), true);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::compress()
  {
    Mat A_temp(this->size(0), this->size(1));
    A_temp.assign(A);
    A.swap(A_temp);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::disp(uint precision) const
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
  inline void uBlasMatrix<ublas_dense_matrix>::init(const GenericSparsityPattern& sparsity_pattern)
  {
    init(sparsity_pattern.size(0), sparsity_pattern.size(1));
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  inline void uBlasMatrix<Mat>::init(const GenericSparsityPattern& sparsity_pattern)
  {
    init(sparsity_pattern.size(0), sparsity_pattern.size(1));

    // Reserve space for non-zeroes
    A.reserve(sparsity_pattern.numNonZero());

    //const SparsityPattern& spattern = dynamic_cast<const SparsityPattern&>(sparsity_pattern);
    const SparsityPattern* pattern_pointer = dynamic_cast<const SparsityPattern*>(&sparsity_pattern);
    if (not pattern_pointer)
      error("Cannot convert GenericSparsityPattern to concrete SparsityPattern type. Aborting.");
    const std::vector< std::set<int> >& pattern = pattern_pointer->pattern();

    std::vector< std::set<int> >::const_iterator set;
    std::set<int>::const_iterator element;
    for(set = pattern.begin(); set != pattern.end(); ++set)
      for(element = set->begin(); element != set->end(); ++element)
        A.push_back(set - pattern.begin(), *element, 0.0);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  inline LogStream& operator<< (LogStream& stream, const uBlasMatrix<Mat>& B)
  {
    // Check if matrix has been defined
    if ( B.size(0) == 0 || B.size(1) == 0 )
    {
      stream << "[ uBlasMatrix matrix (empty) ]";
      return stream;
    }

    uint M = B.size(0);
    uint N = B.size(1);
    stream << "[ uBlasMatrix matrix of size " << M << " x " << N << " ]";

    return stream;
  }
  //-----------------------------------------------------------------------------

}

#endif
