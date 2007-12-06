#include <dolfin/uBlasMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/uBlasFactory.h>

using namespace dolfin;

GenericMatrix* dolfin::uBlasFactory::createMatrix() const 
{
  return new uBlasMatrix<ublas_dense_matrix>(); 
}

SparsityPattern* dolfin::uBlasFactory::createPattern() const 
{
  return new SparsityPattern(); 
}

GenericVector* dolfin::uBlasFactory::createVector() const 
{
  return new uBlasVector(); 
}

dolfin::uBlasFactory dolfin::uBlasFactory::ublasfactory;
