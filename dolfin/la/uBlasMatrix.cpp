#include "ublas.h"
#include "uBlasFactory.h"
#include "uBlasMatrix.h"

namespace dolfin 
{

template <> 
 LinearAlgebraFactory& uBlasMatrix<ublas_sparse_matrix>::factory() const
 { 
   return uBlasFactory::instance(); 
 }


template <> 
 dolfin::LinearAlgebraFactory& dolfin::uBlasMatrix<ublas_dense_matrix>::factory() const
 { 
   return dolfin::uBlasFactory::instance(); 
 }

}
