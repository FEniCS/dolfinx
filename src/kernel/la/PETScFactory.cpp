#ifdef HAVE_PETSC_H

#include <dolfin/SparsityPattern.h>
#include <dolfin/PETScMatrix.h>
#include <dolfin/PETScVector.h>
#include <dolfin/PETScFactory.h>

using namespace dolfin;

GenericSparsityPattern* PETScFactory::createPattern() const 
{
  return new SparsityPattern(); 
}

GenericMatrix* PETScFactory::createMatrix() const 
{ 
  PETScMatrix* pm = new PETScMatrix();
  return pm;
}

GenericVector* PETScFactory:: createVector() const 
{ 
  return new PETScVector(); 
}

PETScFactory PETScFactory::petscfactory;

#endif
