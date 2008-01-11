// Copyright (C) 2004-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2007.
//
// First added:  2004
// Last changed: 2007-12-12

// FIXME: Insert dolfin_assert() where appropriate

#ifdef HAVE_PETSC_H

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScVector.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/MPI.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScVector::PETScVector()
  : GenericVector(), 
    Variable("x", "a sparse vector"),
    x(0), _copy(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(uint N)
  : GenericVector(), 
    Variable("x", "a sparse vector"), 
    x(0), _copy(false)
{
  // Create PETSc vector
  init(N);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(Vec x)
  : GenericVector(),
    Variable("x", "a vector"),
    x(x), _copy(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v)
  : GenericVector(), 
    Variable("x", "a vector"),
    x(0), _copy(false)
{
  *this = v;
}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  clear();
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
  
  if (x)
  {
    const uint n = this->size();
    if (n == N)
    {
      VecZeroEntries(x);
      return;      
    }
  }
  else
    clear();

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
PETScVector* PETScVector::create() const
{
  return new PETScVector();
}
//-----------------------------------------------------------------------------
PETScVector* PETScVector::copy() const
{
  // Not yet implemented
  error("Not yet implemented.");

  return 0;
}
//-----------------------------------------------------------------------------
void PETScVector::axpy(const real a, const PETScVector& x)
{
  VecAXPY(this->x, a, x.vec());
}
//-----------------------------------------------------------------------------
void PETScVector::div(const PETScVector& x)
{
  VecPointwiseDivide(this->x, this->x, x.vec());
  apply();
}
//-----------------------------------------------------------------------------
void PETScVector::mult(const PETScVector& x)
{
  VecPointwiseMult(this->x, this->x, x.vec());
  apply();
}
//-----------------------------------------------------------------------------
void PETScVector::mult(const real a)
{
  dolfin_assert(x);
  VecScale(x, a);
}
//-----------------------------------------------------------------------------
void PETScVector::get(real* values) const
{
  const real* xx = array();
  for (uint i = 0; i < size(); i++)
    values[i] = xx[i];
  restore(xx);
}
//-----------------------------------------------------------------------------
void PETScVector::set(real* values)
{
  real* xx = array();
  for (uint i = 0; i < size(); i++)
    xx[i] = values[i];
  restore(xx);
}
//-----------------------------------------------------------------------------
void PETScVector::add(real* values)
{
  real* xx = array();
  for (uint i = 0; i < size(); i++)
    xx[i] += values[i];
  restore(xx);
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
void PETScVector::clear()
{
  if ( x && !_copy )
  {
    VecDestroy(x);
  }

  x = 0;
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
real* PETScVector::array()
{
  dolfin_assert(x);

  real* data = 0;
  VecGetArray(x, &data);
  dolfin_assert(data);

  return data;
}
//-----------------------------------------------------------------------------
const real* PETScVector::array() const
{
  dolfin_assert(x);

  real* data = 0;
  VecGetArray(x, &data);

  return data;
}
//-----------------------------------------------------------------------------
void PETScVector::restore(const real data[]) const
{
  dolfin_assert(x);

  // Cast away the constness and trust PETSc to do the right thing
  real* tmp = const_cast<real *>(data);
  VecRestoreArray(x, &tmp);
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (const PETScVector& x)
{
  if ( !x.x )
  {
    clear();
    return *this;
  }

  init(x.size());
  VecCopy(x.x, this->x);

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (const real a)
{
  dolfin_assert(x);
  VecSet(x, a);

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator+= (const PETScVector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  const real a = 1.0;
  VecAXPY(this->x, a, x.x);

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator-= (const PETScVector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  const real a = -1.0;
  VecAXPY(this->x, a, x.x);

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
real PETScVector::operator*(const PETScVector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  real a;
  VecDot(x.x, this->x, &a);

  return a;
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
real PETScVector::sum() const
{
  dolfin_assert(x);
  real value = 0.0;
  VecSum(x, &value);
  
  return value;
}
//-----------------------------------------------------------------------------
void PETScVector::disp(uint precision) const
{
  VecView(x, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const PETScVector& x)
{
  // Check if matrix has been defined
  if ( !x.x )
  {
    stream << "[ PETSc vector (empty) ]";
    return stream;
  }
  stream << "[ PETSc vector of size " << x.size() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
void PETScVector::gather(PETScVector& x1, PETScVector& x2, VecScatter& x1sc)
{
  VecScatterBegin(x1sc, x1.vec(), x2.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(x1sc, x1.vec(), x2.vec(), INSERT_VALUES, SCATTER_FORWARD);
}
//-----------------------------------------------------------------------------
void PETScVector::scatter(PETScVector& x1, PETScVector& x2,
			   VecScatter& x1sc)
{
  VecScatterBegin(x1sc, x2.vec(), x1.vec(), INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(x1sc, x2.vec(), x1.vec(), INSERT_VALUES, SCATTER_REVERSE);
}
//-----------------------------------------------------------------------------
VecScatter* PETScVector::createScatterer(PETScVector& x1, PETScVector& x2,
					  const int offset, const int size)
{
  VecScatter* sc = new VecScatter;
  IS* is = new IS;

  ISCreateBlock(MPI_COMM_WORLD, size, 1, &offset, is);
  VecScatterCreate(x1.vec(), PETSC_NULL, x2.vec(), *is, sc);

  return sc;
}
//-----------------------------------------------------------------------------
void PETScVector::fromArray(const real u[], PETScVector& x,
                            uint offset, uint size)
{
  // Workaround to interface PETScVector and arrays

  real* vals = x.array();
  for(uint i = 0; i < size; i++)
    vals[i] = u[i + offset];
  x.restore(vals);
}
//-----------------------------------------------------------------------------
void PETScVector::toArray(real y[], const PETScVector& x,
                          uint offset, uint s)
{
  // Workaround to interface PETScVector and arrays

  const real* vals = x.array();
  for(uint i = 0; i < s; i++)
    y[offset + i] = vals[i];
  x.restore(vals);
}
//-----------------------------------------------------------------------------
void PETScVector::copy(const PETScVector& y, uint off1, uint off2, uint len)
{
  // FIXME: Use gather/scatter for parallel case

  real* xvals = array();
  const real* yvals = y.array();
  for(uint i = 0; i < len; i++)
    xvals[i + off1] = yvals[i + off2];
  restore(xvals);
  restore(yvals);
}
//-----------------------------------------------------------------------------
void PETScVector::copy(const uBlasVector& y, uint off1, uint off2, uint len)
{
  // FIXME: Verify if there's a more efficient implementation

  real* vals = array();
  for(uint i = 0; i < len; i++)
    vals[i + off1] = y[i + off2];
  restore(vals);
}
//-----------------------------------------------------------------------------
Vec PETScVector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
PETScFactory& PETScVector::factory() const
{
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------

#endif
