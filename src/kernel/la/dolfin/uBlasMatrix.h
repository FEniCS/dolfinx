// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-07-05
// Last changed: 2006-07-07

#ifndef __UBLAS_MATRIX_H
#define __UBLAS_MATRIX_H

#include <sstream>
#include <iomanip>
#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/Variable.h>
#include <dolfin/GenericMatrix.h>
#include <dolfin/ublas.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/uBlasLUSolver.h>

namespace dolfin
{

  /// This class represents a matrix (dense or sparse) of dimension M x N.
  /// It is a wrapper for a Boost uBLAS matrix of type Mat.
  ///
  /// The interface is intended to provide uniformily with respect to other
  /// matrix data types. For advanced usage, refer to the documentation for 
  /// uBLAS which can be found at 
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.

  /// Developer note: specialised member functions must be inlined to avoid link errors.
  template< class Mat > 
  class uBlasMatrix; 

  template< class Mat > 
  LogStream& operator<<  (LogStream&, const uBlasMatrix<Mat>&);


  template< class Mat >
  class uBlasMatrix : public Variable, 
                      public GenericMatrix,
		                  public Mat
  {
  public:
    
    /// Constructor
    uBlasMatrix();
    
    /// Constructor
    uBlasMatrix(const uint M, const uint N);

    /// Destructor
    ~uBlasMatrix();

    /// Assignment from a matrix_expression
    template <class E>
    uBlasMatrix<Mat>& operator=(const ublas::matrix_expression<E>& A) const
    { 
      Mat::operator=(A); 
      return *this;
    } 

    /// Return number of rows (dim = 0) or columns (dim = 1) 
    uint size(uint dim) const;

    /// Access element value
    real get(uint i, uint j) const;

    /// Set element value
    void set(uint i, uint j, real value);

    /// Get non-zero values of row i
    void getRow(const uint i, int& ncols, Array<int>& columns, Array<real>& values) const;

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

    /// Set given rows to identity matrix
    void ident(const int rows[], int m);

    /// Compute product y = Ax
    void mult(const uBlasVector& x, uBlasVector& y) const;

    /// Display matrix
    void disp(const uint precision = 2) const;

    /// The below functions have specialisations for particular matrix types.
    /// In order to link correctly, they must be made inline functions.

    /// Initialize M x N matrix
    void init(uint M, uint N);

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    void init(uint M, uint N, uint nzmax);

    /// Set block of values. The function apply() must be called to commit changes.
    void set(const real block[], const int rows[], int m, 
                            const int cols[], int n);

    /// Add block of values. The function apply() must be called to commit changes.
    void add(const real block[], const int rows[], int m, 
                            const int cols[], int n);

    /// Return average number of non-zeros per row
    uint nzmax() const;

    friend LogStream& operator<< <Mat> (LogStream&, const uBlasMatrix<Mat>&);

  private:

    /// Matrix used internally for assembly of sparse matrices
    ublas_assembly_matrix Assembly_matrix;

    /// Matrix state
    bool assembled;

  };

  //---------------------------------------------------------------------------
  // Implementation of uBlasMatrix
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>::uBlasMatrix() : assembled(true)
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>::uBlasMatrix(const uint M, const uint N) : assembled(true)
  { 
    init(M,N); 
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uBlasMatrix<Mat>::~uBlasMatrix()
  { 
    // Do nothing 
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  uint uBlasMatrix<Mat>::size(const uint dim) const
  {
    dolfin_assert( dim < 2 );
    return (dim == 0 ? this->size1() : this->size2());  
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  inline real uBlasMatrix<Mat>::get(uint i, uint j) const
  { 
    return (*this)(i, j);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  inline void uBlasMatrix<Mat>::set(uint i, uint j, real value) 
  { 
    (*this)(i, j) = value;
  }
  //---------------------------------------------------------------------------
  template < class Mat >  
  void uBlasMatrix< Mat >::getRow(const uint i, int& ncols, Array<int>& columns, 
				  Array<real>& values) const
  {
    if( !assembled )
      dolfin_error("Matrix has not been assembled. Did you forget to call A.apply()?"); 

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
  void uBlasMatrix<Mat>::lump(uBlasVector& m) const
  {
    if( !assembled )
      dolfin_error("Matrix has not been assembled. Did you forget to call A.apply()?"); 

    const uint n = this->size(1);
    m.init( n );
    ublas::scalar_vector<double> one(n, 1.0);
    ublas::axpy_prod(*this, one, m, true);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::solve(uBlasVector& x, const uBlasVector& b) const
  {    
    if( !assembled )
      dolfin_error("Matrix has not been assembled. Did you forget to call A.apply()?"); 

    uBlasLUSolver solver;
    solver.solve(*this, x, b);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::invert()
  {
    if( !assembled )
      dolfin_error("Matrix has not been assembled. Did you forget to call A.apply()?"); 

    uBlasLUSolver solver;
    solver.invert(*this);
  }
//-----------------------------------------------------------------------------
  template <class Mat>
  void uBlasMatrix<Mat>::apply()
  {
    // Assign temporary assembly matrix to the sparse matrix
    if( !assembled )
    {
      // Assign temporary assembly matrix to the matrix
      this->assign(Assembly_matrix);
      assembled = true;

      // Free memory
      Assembly_matrix.resize(0,0, false);
    } 
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::zero()
  {
    if( !assembled )
      dolfin_error("Matrix has not been assembled. Did you forget to call A.apply()?"); 

    // Clear destroys non-zero structure of a sparse matrix 
    this->clear();

    // Set all non-zero values to zero without detroying non-zero pattern
  //  (*this) *= 0.0;
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::ident(const int rows[], const int m) 
  {
    if( !assembled )
      dolfin_error("Matrix has not been assembled. Did you forget to call A.apply()?"); 

    const uint n = this->size(1);
    for(int i = 0; i < m; ++i)
      ublas::row(*this, rows[i]) = ublas::unit_vector<double> (n, rows[i]);
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::mult(const uBlasVector& x, uBlasVector& y) const
  {
    if( !assembled )
      dolfin_error("Matrix has not been assembled. Did you forget to call A.apply()?"); 

    ublas::axpy_prod(*this, x, y, true);
  }
  //-----------------------------------------------------------------------------
  template <class Mat>  
  void uBlasMatrix<Mat>::disp(const uint precision) const
  {
    typename Mat::const_iterator1 it1;  // Iterator over rows
    typename Mat::const_iterator2 it2;  // Iterator over entries

    for (it1 = this->begin1(); it1 != this->end1(); ++it1)
    {
      std::stringstream line;
      line << std::setiosflags(std::ios::scientific);
      line << std::setprecision(precision);
    
      line << "|";
      for (it2 = it1.begin(); it2 != it1.end(); ++it2)
        line << " (" << it2.index1() << ", " << it2.index2() << ", " << *it2 << ")";
      line << " |";

      dolfin::cout << line.str().c_str() << dolfin::endl;
    }  
  }
  //-----------------------------------------------------------------------------
  // Specialised member functions (must be inlined to avoid link errors)
  //-----------------------------------------------------------------------------
  template <> 
  inline void uBlasMatrix< ublas_dense_matrix >::init(const uint M, const uint N)
  {
    // Resize matrix
    if( size(0) != M || size(1) != N )
      this->resize(M, N, false);  
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  inline void uBlasMatrix<Mat>::init(const uint M, const uint N)
  {
    // Resize matrix
    if( size(0) != M || size(1) != N )
      this->resize(M, N, false);  

    // Resize assembly matrix
    if(Assembly_matrix.size1() != M && Assembly_matrix.size2() != N )
      Assembly_matrix.resize(M, N, false);
  }
  //---------------------------------------------------------------------------
  template <> 
  inline void uBlasMatrix<ublas_dense_matrix>::init(const uint M, const uint N, 
                                                    const uint nzmax)
  {
    init(M, N);
  }
  //---------------------------------------------------------------------------
  template <class Mat> 
  inline void uBlasMatrix<Mat>::init(const uint M, const uint N, const uint nzmax)
  {
    init(M, N);

    // Reserve space for non-zeroes
    const uint total_nz = nzmax*size(0);
    this->reserve(total_nz);
  }
  //---------------------------------------------------------------------------
  template <>  
  inline void uBlasMatrix<ublas_dense_matrix>::set(const real block[], 
                 const int rows[], const int m, const int cols[], const int n)
  {
   for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        (*this)(rows[i] , cols[j]) = block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <>  
  inline void uBlasMatrix<ublas_dense_matrix>::add(const real block[], 
                 const int rows[], const int m, const int cols[], const int n)
  {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        (*this)(rows[i] , cols[j]) += block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  inline void uBlasMatrix<Mat>::add(const real block[], const int rows[], 
                                    const int m, const int cols[], const int n)
  {
    if( assembled )
    {
      Assembly_matrix.assign(*this);
      assembled = false; 
    }
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Assembly_matrix(rows[i] , cols[j]) += block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  inline void uBlasMatrix<Mat>::set(const real block[], const int rows[], 
                                    const int m, const int cols[], const int n)
  {
    if( assembled )
    {
      Assembly_matrix.assign(*this);
      assembled = false; 
    }
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        Assembly_matrix(rows[i] , cols[j]) = block[i*n + j];
  }
  //---------------------------------------------------------------------------
  template <>  
  inline uint uBlasMatrix<ublas_dense_matrix>::nzmax() const 
  { 
    return 0; 
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  inline uint uBlasMatrix<Mat>::nzmax() const 
  { 
    return this->nnz()/size(0); 
  }
  //---------------------------------------------------------------------------
  template <class Mat>  
  inline LogStream& operator<< (LogStream& stream, const uBlasMatrix< Mat >& A)
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
