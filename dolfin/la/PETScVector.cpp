// Copyright (C) 2004-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2008.
// Modified by Martin Sandve Alnes 2008
//
// First added:  2004
// Last changed: 2008-12-27

#ifdef HAS_PETSC

#include <cmath>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScVector.h"
#include "uBLASVector.h"
#include "PETScFactory.h"
#include <dolfin/main/MPI.h>

namespace dolfin
{
  class PETScVectorDeleter
  {
  public:
    void operator() (Vec* x)
    {
      if (x)
        VecDestroy(*x);
      delete x;
    }
  };
}

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScVector::PETScVector():
    Variable("x", "a sparse vector"),
    x(static_cast<Vec*>(0), PETScVectorDeleter())
{
 // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(uint N):
    Variable("x", "a sparse vector"),
    x(static_cast<Vec*>(0), PETScVectorDeleter())
{
  // Create PETSc vector
  resize(N);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(boost::shared_ptr<Vec> x):
    Variable("x", "a vector"),
    x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v):
    Variable("x", "a vector"),
    x(static_cast<Vec*>(0), PETScVectorDeleter())
{
  *this = v;
}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  // Do nothing. The custom shared_ptr deleter takes care of the cleanup.
}
//-----------------------------------------------------------------------------
void PETScVector::resize(uint N)
{
  if (x && this->size() == N)
    return;

  // Create vector
  if (!x.unique())
    error("Cannot resize PETScVector. More than one object points to the underlying PETSc object.");
  boost::shared_ptr<Vec> _x(new Vec, PETScVectorDeleter());
  x = _x;

  if (MPI::num_processes() > 1)
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, x.get());
  else
    VecCreate(PETSC_COMM_SELF, x.get());

  // Set size
  VecSetSizes(*x, PETSC_DECIDE, N);
  VecSetFromOptions(*x);
}
//-----------------------------------------------------------------------------
PETScVector* PETScVector::copy() const
{
  PETScVector* v = new PETScVector(*this);
  return v;
}
//-----------------------------------------------------------------------------
void PETScVector::get(double* values) const
{
  dolfin_assert(x);

  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecGetValues(*x, m, rows, values);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::set(double* values)
{
  dolfin_assert(x);

  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecSetValues(*x, m, rows, values, INSERT_VALUES);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::add(double* values)
{
  dolfin_assert(x);

  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecSetValues(*x, m, rows, values, ADD_VALUES);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::get(double* block, uint m, const uint* rows) const
{
  dolfin_assert(x);
  VecGetValues(*x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block);
}
//-----------------------------------------------------------------------------
void PETScVector::set(const double* block, uint m, const uint* rows)
{
  dolfin_assert(x);
  VecSetValues(*x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block,
               INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::add(const double* block, uint m, const uint* rows)
{
  dolfin_assert(x);
  VecSetValues(*x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block,
               ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::apply()
{
  dolfin_assert(x);
  VecAssemblyBegin(*x);
  VecAssemblyEnd(*x);
}
//-----------------------------------------------------------------------------
void PETScVector::zero()
{
  dolfin_assert(x);
  double a = 0.0;
  VecSet(*x, a);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScVector::size() const
{
  int n = 0;
  if (x)
    VecGetSize(*x, &n);
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

  // Check for self-assignment
  if (this != &v)
  {
    resize(v.size());
    VecCopy(*(v.x), *x);
  }
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (double a)
{
  dolfin_assert(x);
  VecSet(*x, a);
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
const PETScVector& PETScVector::operator*= (const double a)
{
  dolfin_assert(x);
  VecScale(*x, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator*= (const GenericVector& y)
{
  dolfin_assert(x);
  
  const PETScVector& v = y.down_cast<PETScVector>();
  dolfin_assert(v.x);

  if (size() != v.size())
    error("The vectors must be of the same size.");  

  VecPointwiseMult(*x,*x,*v.x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator/= (const double a)
{
  dolfin_assert(x);
  dolfin_assert(a != 0.0);

  const double b = 1.0/a;
  VecScale(*x, b);
  return *this;
}
//-----------------------------------------------------------------------------
double PETScVector::inner(const GenericVector& y) const
{
  dolfin_assert(x);

  const PETScVector& v = y.down_cast<PETScVector>();
  dolfin_assert(v.x);

  double a;
  VecDot(*(v.x), *x, &a);
  return a;
}
//-----------------------------------------------------------------------------
void PETScVector::axpy(double a, const GenericVector& y)
{
  dolfin_assert(x);

  const PETScVector& v = y.down_cast<PETScVector>();
  dolfin_assert(v.x);

  if (size() != v.size())
    error("The vectors must be of the same size.");

  VecAXPY(*x, a, *(v.x));
}
//-----------------------------------------------------------------------------
double PETScVector::norm(std::string norm_type) const
{
  dolfin_assert(x);

  double value = 0.0;
  if (norm_type == "l1")
    VecNorm(*x, NORM_1, &value);
  else if (norm_type == "l2")
    VecNorm(*x, NORM_2, &value);
  else if (norm_type == "linf")
    VecNorm(*x, NORM_INFINITY, &value);
  else
    error("Norm type for PETScVector unknown.");

  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::min() const
{
  dolfin_assert(x);

  double value = 0.0;
  int position = 0;
  VecMin(*x, &position, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::max() const
{
  dolfin_assert(x);

  double value = 0.0;
  int position = 0;
  VecMax(*x, &position, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::sum() const
{
  dolfin_assert(x);

  double value = 0.0;
  VecSum(*x, &value);
  return value;
}
//-----------------------------------------------------------------------------
void PETScVector::disp(uint precision) const
{
  VecView(*x, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Vec> PETScVector::vec() const
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
