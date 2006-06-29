// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-23
// Last changed: 

#ifndef __UBLAS_H
#define __UBLAS_H

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

// These two files must be included due to a bug in Boost version < 1.33.
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


namespace dolfin
{
  /// Various typedefs for uBlas data types 

  namespace ublas = boost::numeric::ublas;

  // uBlas dense matrix
  typedef ublas::matrix<double> ublas_matrix;

  // Set type of underlying uBlas sparse matrix. 
  typedef ublas::compressed_matrix<double> ublas_sparse_matrix;
//  typedef ublas::generalized_vector_of_vector< double, ublas::row_major, 
//            ublas::vector<ublas::compressed_vector<double> > > ublas_sparse_matrix;


  // uBlas sparse matrix for temporoary assembly
  typedef ublas::generalized_vector_of_vector< double, ublas::row_major, 
            ublas::vector<ublas::compressed_vector<double> > > ublas_assembly_matrix;


  // uBlas column-major compressed matrix
  typedef ublas::compressed_matrix<double,ublas::column_major> ublas_sparse_matrix_cmajor;

  // uBlas vector
  typedef ublas::vector<double> ublas_vector;
  typedef ublas::vector_range<ublas_vector> ublas_vector_range;

  // uBlas dense matrix (column major format)
  typedef ublas::matrix<double,ublas::column_major> ublas_matrix_cmajor;
  typedef ublas::matrix_range<ublas_matrix_cmajor> ublas_matrix_cmajor_range;

  // uBlas column major upper triangular matrix
  typedef ublas::triangular_matrix<double, ublas::upper, ublas::column_major> ublas_matrix_tri;
  typedef ublas::matrix_range<ublas_matrix_tri> ublas_matrix_range_tri;
  typedef ublas::matrix_column<ublas_matrix_tri> ublas_matrix_col_tri;

}

#endif
