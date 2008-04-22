// Copyright (C) 2006-2007 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006-2007.
// Modified by Ola Skavhaug 2007-2008.
// Modified by Kent-Andre Mardal 2008.
//
// First added:  2006-07-05
// Last changed: 2008-04-08

#ifndef __UBLAS_MATRIX_H
#define __UBLAS_MATRIX_H

#include <sstream>
#include <iomanip>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>
#include "GenericMatrix.h"
#include "GenericSparsityPattern.h"
#include "SparsityPattern.h"
#include "ublas.h"
#include "uBlasVector.h"
#include "uBlasLUSolver.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{

  /// This class represents a matrix (dense or sparse) of dimension M x N.
  /// It is a wrapper for a Boost uBLAS matrix of type Mat.
  ///
  /// The interface is intended to provide uniformity with respect to other
  /// matrix data types. For advanced usage, refer to the documentation for 
  /// uBLAS which can be found at 
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.

  /// Developer note: specialised member functions must be inlined to avoid link errors.

  template< class Mat >
  class uBlasMatrix : public Variable, 
                      public GenericMatrix,
                      public Mat
  {
  public:
    
    /// Constructor
    uBlasMatrix();
    
    /// Constructor
    uBlasMatrix(uint M, uint N);

   /// Constructor from a uBlas matrix_expression
    template <class E>
    uBlasMatrix(const ublas::matrix_expression<E>& A) : Mat(A) {}

    /// Destructor
    ~uBlasMatrix();

    /// Assignment from a matrix_expression
    template <class E>
    uBlasMatrix<Mat>& operator=(const ublas::matrix_expression<E>& A)
    { 
      Mat::operator=(A); 
      return *this;
    } 

    /// Return number of rows (dim = 0) or columns (dim = 1) 
    uint size(uint dim) const;

    /// Set block of values
    void set(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add block of values
    void add(const real* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Get non-zero values of row i
    void getrow(uint i, int& ncols, Array<int>& columns, Array<real>& values) const;

    /// Get block of values
    void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const;

    /// Lump matrix into vector m
    void lump(uBlasVector& m) const;

    /// Solve Ax = b out-of-place (A is not destroyed)
    void solve(uBlasVector& x, const uBlasVector& b) const;

    /// Compute inverse of matrix
    void invert();

    /// Apply changes to matrix 
    void apply();

    /// Set all entries to zero
    void zero();

    /// Set given rows to zero matrix
    void zero(uint m, const uint* rows);

    /// Set given rows to identity matrix
    void ident(uint m, const uint* rows);

    /// Compute product y = Ax
    void mult(const uBlasVector& x, uBlasVector& y) const;

    /// Compute product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const; 

    /// Compress matrix (eliminate all non-zeros from a sparse matrix) 
    void compress();

    /// Display matrix
    void disp(uint precision = 2) const;

    /// The below functions have specialisations for particular matrix types.
    /// In order to link correctly, they must be made inline functions.

    /// Return concrete (const) uBlasMatrix<Mat> instance
    virtual const uBlasMatrix<Mat>* instance() const {
      return this;
    }

    /// Return concrete uBlasMatrix<Mat> instance
    virtual uBlasMatrix<Mat>* instance() {
      return this;
    }

    /// Initialize M x N matrix
    void init(uint M, uint N);

    /// Initialize a matrix from the sparsity pattern
    void init(const GenericSparsityPattern& sparsity_pattern);

    /// Create uninitialized matrix
    uBlasMatrix<Mat>* create() const;

    /// Create copy of matrix
    uBlasMatrix<Mat>* copy() const;

    LinearAlgebraFactory& factory() const;


    //friend LogStream& operator<< <Mat> (LogStream&, const uBlasMatrix<Mat>&);

  private:

  };


  //---------------------------------------------------------------------------
  // Implementation of uBlasMatrix
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>::uBlasMatrix()
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>::uBlasMatrix(uint M, uint N) : Mat(M,N)
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
      this->resize(M, N, false);  

    // Clear matrix (detroys any structure)
    this->clear();
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>* uBlasMatrix<Mat>::create() const
  {
    return new uBlasMatrix<Mat>();
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
    return (dim == 0 ? this->size1() : this->size2());  
  }
  //---------------------------------------------------------------------------
  template < class Mat >  
  void uBlasMatrix< Mat >::getrow(uint i, int& ncols, Array<int>& columns, 
				  Array<real>& values) const
  {
    // Reference to matrix row (throw away const-ness and trust uBlas)
    ublas::matrix_row< uBlasMatrix<Mat> > row( *(const_cast< uBlasMatrix<Mat>* >(this)) , i);

    typename ublas::matrix_row< uBlasMatrix<Mat> >::const_iterator component;

    // Insert values into Arrays
    columns.clear();
    values.clear();
    for (component=row.begin(); component != row.end(); ++component) 
    {
      columns.push_back( component.index() );
      values.push_back( *component );
    }
    ncols = columns.size();
  }
  //-----------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix<Mat>::set(const real* block,
                                    uint m, const uint* rows,
                                    uint n, const uint* cols)
  {
    for (uint i = 0; i < m; i++)
      for (uint j = 0; j < n; j++)
        (*this)(rows[i] , cols[j]) = block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::add(const real* block,
                                    uint m, const uint* rows,
                                    uint n, const uint* cols)
  {
    for (uint i = 0; i < m; i++)
      for (uint j = 0; j < n; j++)
        (*this)(rows[i] , cols[j]) += block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix<Mat>::get(real* block,
                                    uint m, const uint* rows,
                                    uint n, const uint* cols) const
  {
    for(uint i = 0; i < m; ++i)
      for(uint j = 0; j < n; ++j)
        block[i*n + j] = (*this)(rows[i], cols[j]);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::lump(uBlasVector& m) const
  {
    const uint n = this->size(1);
    m.init( n );
    ublas::scalar_vector<double> one(n, 1.0);
    ublas::axpy_prod(*this, one, m.vec(), true);
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
    solver.invert(*this);
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
    (*this) *= 0.0;
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::zero(uint m, const uint* rows) 
  {
    for(uint i = 0; i < m; ++i) {
      ublas::row(*this, rows[i]) *= 0.0;  
    }
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::ident(uint m, const uint* rows) 
  {
    const uint n = this->size(1);
    for(uint i = 0; i < m; ++i)
      ublas::row(*this, rows[i]) = ublas::unit_vector<double> (n, rows[i]);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::mult(const uBlasVector& x, uBlasVector& y) const
  {
    ublas::axpy_prod(*this, x.vec(), y.vec(), true);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::mult(const GenericVector& x_, GenericVector& y_, bool transposed) const
  {
    const uBlasVector* x = dynamic_cast<const uBlasVector*>(x_.instance());  
    if (!x)  
      error("The first vector needs to be of type uBlasVector"); 

    uBlasVector* y = dynamic_cast<uBlasVector*>(y_.instance());  
    if (!y)  
      error("The second vector needs to be of type uBlasVector"); 

    if (transposed==true) error("The transposed version of the uBLAS matrix vector product is not yet implemented");  
    this->mult(*x, *y); 
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::compress()
  {
    Mat A_temp(this->size(0), this->size(1));
    A_temp.assign(*this);
    this->swap(A_temp);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::disp(uint precision) const
  {
    typename Mat::const_iterator1 it1;  // Iterator over rows
    typename Mat::const_iterator2 it2;  // Iterator over entries

    for (it1 = this->begin1(); it1 != this->end1(); ++it1)
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
    this->reserve(sparsity_pattern.numNonZero());

    //const SparsityPattern& spattern = dynamic_cast<const SparsityPattern&>(sparsity_pattern);
    const SparsityPattern* pattern_pointer = dynamic_cast<const SparsityPattern*>(&sparsity_pattern);
    if (not pattern_pointer)
      error("Cannot convert GenericSparsityPattern to concrete SparsityPattern type. Aborting.");
    const std::vector< std::set<int> >& pattern = pattern_pointer->pattern();

    std::vector< std::set<int> >::const_iterator set;
    std::set<int>::const_iterator element;
    for(set = pattern.begin(); set != pattern.end(); ++set)
      for(element = set->begin(); element != set->end(); ++element)
        this->push_back(set - pattern.begin(), *element, 0.0);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  inline LogStream& operator<< (LogStream& stream, const uBlasMatrix<Mat>& A)
  {
    // Check if matrix has been defined
    if ( A.size(0) == 0 || A.size(1) == 0 )
    {
      stream << "[ uBlasMatrix matrix (empty) ]";
      return stream;
    }

    uint M = A.size(0);
    uint N = A.size(1);
    stream << "[ uBlasMatrix matrix of size " << M << " x " << N << " ]";

    return stream;
  }
  //-----------------------------------------------------------------------------

}

#endif
