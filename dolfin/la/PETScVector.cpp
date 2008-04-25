// Copyright (C) 2004-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2007.
// Modified by Martin Alnæs 2008
//
// First added:  2004
// Last changed: 2008-04-23

// FIXME: Insert dolfin_assert() where appropriate

#ifdef HAS_PETSC

#include <cmath>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScVector.h"
#include "uBlasVector.h"
#include "PETScFactory.h"
#include <dolfin/main/MPI.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScVector::PETScVector():
    Variable("x", "a sparse vector"),
    x(0), _copy(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(uint N):
    Variable("x", "a sparse vector"), 
    x(0), _copy(false)
{
  // Create PETSc vector
  init(N);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(Vec x):
    Variable("x", "a vector"),
    x(x), _copy(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v):
    Variable("x", "a vector"),
    x(0), _copy(false)
{
  *this = v;
}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  if (x && !_copy)
    VecDestroy(x);
}
//-----------------------------------------------------------------------------
void PETScVector::init(uint N)
{
  // Two cases:
  //
  //   1. Already allocated and dimension changes -> reallocate
  //   2. Not allocated -> allocate
  //
  // Otherwise do nothing
  
  if (x && this->size() == N)
  {
    VecZeroEntries(x);
    return;      
  }
  else
  {
    if (x && !_copy)
      VecDestroy(x);
  }

  // Create vector
  if (MPI::numProcesses() > 1)
  {
    dolfin_debug("PETScVector::init(N) - VecCreateMPI");
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &x);
  }
  else
    VecCreate(PETSC_COMM_SELF, &x);

  VecSetSizes(x, PETSC_DECIDE, N);
  VecSetFromOptions(x);

  // Set all entries to zero
  PetscScalar a = 0.0;
  VecSet(x, a);
}
//-----------------------------------------------------------------------------
PETScVector* PETScVector::copy() const
{
  PETScVector* v = new PETScVector(*this); 
  return v; 
}
//-----------------------------------------------------------------------------
void PETScVector::get(real* values) const
{
  dolfin_assert(x);
  
  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecGetValues(x, m, rows, values);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::set(real* values)
{
  dolfin_assert(x);
  
  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecSetValues(x, m, rows, values, INSERT_VALUES);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::add(real* values)
{
  dolfin_assert(x);
  
  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecSetValues(x, m, rows, values, ADD_VALUES);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::get(real* block, uint m, const uint* rows) const
{
  dolfin_assert(x);
  VecGetValues(x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block);
}
//-----------------------------------------------------------------------------
void PETScVector::set(const real* block, uint m, const uint* rows)
{
  dolfin_assert(x);
  VecSetValues(x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block,
               INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::add(const real* block, uint m, const uint* rows)
{
  dolfin_assert(x);
  VecSetValues(x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block,
               ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::apply()
{
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
void PETScVector::zero()
{
  dolfin_assert(x);
  real a = 0.0;
  VecSet(x, a);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScVector::size() const
{
  int n = 0;
  if (x)
    VecGetSize(x, &n);
  
  return static_cast<uint>(n);
}
//-----------------------------------------------------------------------------
const GenericVector& PETScVector::operator= (const GenericVector& v)
{
  *this = v.down_cast<PETScVector>();
  return *this; 
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (const PETScVector& v)
{
  dolfin_assert(v.x);

  init(v.size());
  VecCopy(v.x, x);

  return *this; 
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator+= (const GenericVector& x)
{
  this->axpy(1.0, x); 
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator-= (const GenericVector& x)
{
  this->axpy(-1.0, x); 
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator*= (const real a)
{
  dolfin_assert(x);
  VecScale(x, a);
  
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator/= (const real a)
{
  dolfin_assert(x);
  dolfin_assert(a != 0.0);

  const real b = 1.0 / a;
  VecScale(x, b);
  
  return *this;
}
//-----------------------------------------------------------------------------
real PETScVector::inner(const GenericVector& y) const
{
  dolfin_assert(x);

  const PETScVector& v = y.down_cast<PETScVector>();
  dolfin_assert(v.x);

  real a;
  VecDot(v.x, x, &a);

  return a;
}
//-----------------------------------------------------------------------------
void PETScVector::axpy(real a, const GenericVector& y) 
{
  dolfin_assert(x);

  const PETScVector& v = y.down_cast<PETScVector>();
  dolfin_assert(v.x);

  if (size() != v.size())
    error("The vectors must be of the same size.");  

  VecAXPY(x, a, v.x);
}
//-----------------------------------------------------------------------------
real PETScVector::norm(VectorNormType type) const
{
  dolfin_assert(x);

  real value = 0.0;

  switch (type) {
  case l1:
    VecNorm(x, NORM_1, &value);
    break;
  case l2:
    VecNorm(x, NORM_2, &value);
    break;
  default:
    VecNorm(x, NORM_INFINITY, &value);
  }
  
  return value;
}
//-----------------------------------------------------------------------------
void PETScVector::disp(uint precision) const
{
  VecView(x, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
Vec PETScVector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& PETScVector::factory() const
{
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const PETScVector& x)
{
  stream << "[ PETSc vector of size " << x.size() << " ]";
  return stream;
}
//-----------------------------------------------------------------------------

#endif
