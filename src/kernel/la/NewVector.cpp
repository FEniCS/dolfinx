// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewVector.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewVector::NewVector()
{
  // Initialize PETSc
  PETScManager::init();

  // Don't initialize the vector
  v = 0;
}
//-----------------------------------------------------------------------------
NewVector::NewVector(unsigned int size)
{
  if(size < 0)
    dolfin_error("Size of vector must be non-negative.");

  // Initialize PETSc
  PETScManager::init();

  // Create PETSc vector
  v = 0;
  init(size);
}
//-----------------------------------------------------------------------------
NewVector::NewVector(const Vector &x)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc vector
  v = 0;
  init(x.size());

  unsigned int n = size();

  for(unsigned int i = 0; i < n; i++)
    setvalue(i, x(i));
}
//-----------------------------------------------------------------------------
NewVector::~NewVector()
{
  clear();
}
//-----------------------------------------------------------------------------
void NewVector::init(unsigned int size)
{
  // Two cases:
  //
  //   1. Already allocated and dimension changes -> reallocate
  //   2. Not allocated -> allocate
  //
  // Otherwise do nothing
  
  if(v)
  {
    unsigned int n = this->size();

    if(n == size)
    {
      return;      
    }
  }
  else
  {
    clear();
  }

  VecCreate(PETSC_COMM_WORLD, &v);
  VecSetSizes(v, PETSC_DECIDE, size);
  VecSetFromOptions(v);
}
//-----------------------------------------------------------------------------
void NewVector::setvalue(int i, const real r)
{
  VecSetValue(v, i, r, INSERT_VALUES);

  VecAssemblyBegin(v);
  VecAssemblyEnd(v);
}
//-----------------------------------------------------------------------------
real NewVector::getvalue(int i) const
{
  // Assumes uniprocessor case.

  real val;

  PetscScalar    *array;
  VecGetArray(v, &array);
  val = array[i];
  VecRestoreArray(v, &array);

  return val;
}
//-----------------------------------------------------------------------------
void NewVector::clear()
{
  if(v)
  {
    VecDestroy(v);
  }

  v = 0;
}
//-----------------------------------------------------------------------------
unsigned int NewVector::size() const
{
  int n;

  VecGetSize(v, &n);

  return n;
}
//-----------------------------------------------------------------------------
Vec NewVector::vec()
{
  return v;
}
//-----------------------------------------------------------------------------
const Vec NewVector::vec() const
{
  return v;
}
//-----------------------------------------------------------------------------
NewVector::Index NewVector::operator()(int i)
{
  Index ind(i, *this);

  return ind;
}
//-----------------------------------------------------------------------------
void NewVector::disp() const
{
  VecView(v, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
NewVector::Index::Index(int i, NewVector &v) : i(i), v(v)
{
}
//-----------------------------------------------------------------------------
void NewVector::Index::operator =(const real r)
{
  v.setvalue(i, r);
}
//-----------------------------------------------------------------------------
NewVector::Index::operator real() const
{
  return v.getvalue(i);
}
//-----------------------------------------------------------------------------
