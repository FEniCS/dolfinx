// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.

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
NewVector::NewVector(uint size)
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

  const uint n = size();
  for (uint i = 0; i < n; i++)
    setvalue(i, x(i));
}
//-----------------------------------------------------------------------------
NewVector::~NewVector()
{
  clear();
}
//-----------------------------------------------------------------------------
void NewVector::init(uint size)
{
  // Two cases:
  //
  //   1. Already allocated and dimension changes -> reallocate
  //   2. Not allocated -> allocate
  //
  // Otherwise do nothing
  
  if (v)
  {
    const uint n = this->size();

    if (n == size)
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
void NewVector::add(const real a, const NewVector& x) const
{
  VecAXPY(&a, x.vec(), v);
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
dolfin::uint NewVector::size() const
{
  int n = 0;
  VecGetSize(v, &n);

  return static_cast<uint>(n);
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
real* NewVector::array()
{
  real* data = 0;
  VecGetArray(v, &data);

  return data;
}
//-----------------------------------------------------------------------------
void NewVector::restore(real data[])
{
  VecRestoreArray(v, &data);
}
//-----------------------------------------------------------------------------
NewVector::Index NewVector::operator() (uint i)
{
  Index index(i, *this);

  return index;
}
//-----------------------------------------------------------------------------
const NewVector& NewVector::operator= (real a)
{
  VecSet(&a, v);

  return *this;
}
//-----------------------------------------------------------------------------
void NewVector::disp() const
{
  VecView(v, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
void NewVector::setvalue(uint i, const real r)
{
  VecSetValue(v, static_cast<int>(i), r, INSERT_VALUES);

  VecAssemblyBegin(v);
  VecAssemblyEnd(v);
}
//-----------------------------------------------------------------------------
real NewVector::getvalue(uint i) const
{
  // Assumes uniprocessor case.

  real val = 0.0;

  PetscScalar *array = 0;
  VecGetArray(v, &array);
  val = array[i];
  VecRestoreArray(v, &array);

  return val;
}
//-----------------------------------------------------------------------------
// NewVector::Index
//-----------------------------------------------------------------------------
NewVector::Index::Index(uint i, NewVector &v) : i(i), v(v)
{
  // Do nothing
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
