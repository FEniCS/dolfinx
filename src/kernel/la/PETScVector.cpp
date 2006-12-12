// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2004
// Last changed: 2006-12-12

// FIXME: Insert dolfin_assert() where appropriate

#ifdef HAVE_PETSC_H

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/PETScVector.h>
#include <dolfin/uBlasVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScVector::PETScVector()
  : GenericVector(), 
    Variable("x", "a sparse vector"),
    x(0), _copy(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const uint N)
  : GenericVector(), 
    Variable("x", "a sparse vector"), 
    x(0), _copy(false)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc vector
  init(N);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(Vec x)
  : GenericVector(),
    Variable("x", "a vector"),
    x(x), _copy(true)
{
  // Initialize PETSc 
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v)
  : GenericVector(), 
    Variable("x", "a vector"),
    x(0), _copy(false)
{
  // Initialize PETSc 
  PETScManager::init();

  *this = v;
}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  clear();
}
//-----------------------------------------------------------------------------
void PETScVector::init(const uint N)
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
      return;      
  }
  else
    clear();

  // Create vector
  VecCreate(PETSC_COMM_SELF, &x);
  VecSetSizes(x, PETSC_DECIDE, N);
  VecSetFromOptions(x);

  // Set all entries to zero
  PetscScalar a = 0.0;
  VecSet(x, a);
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
void PETScVector::set(const real block[], const int cols[], const int n)
{
  VecSetValues(x, n, cols, block, INSERT_VALUES); 
}
//-----------------------------------------------------------------------------
void PETScVector::add(const real block[], const int cols[], const int n)
{
  dolfin_assert(x);
  VecSetValues(x, n, cols, block, ADD_VALUES); 
}
//-----------------------------------------------------------------------------
void PETScVector::get(real block[], const int cols[], const int n) const
{
  dolfin_assert(x);
  VecGetValues(x, n, cols, block); 
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
  VecGetSize(x, &n);

  return static_cast<uint>(n);
}
//-----------------------------------------------------------------------------
real PETScVector::get(const uint i) const
{
  dolfin_assert(x);

  // FIXME: Assumes uniprocessor case

  real val = 0.0;

  PetscScalar *array = 0;
  VecGetArray(x, &array);
  val = array[i];
  VecRestoreArray(x, &array);

  return val;
}
//-----------------------------------------------------------------------------
void PETScVector::set(const uint i, const real value)
{
  dolfin_assert(x);

  VecSetValue(x, static_cast<int>(i), value, INSERT_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
void PETScVector::add(const uint i, const real value)
{
  dolfin_assert(x);

  VecSetValue(x, static_cast<int>(i), value, ADD_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
Vec PETScVector::vec() const
{
  return x;
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
void PETScVector::restore(real data[])
{
  dolfin_assert(x);

  VecRestoreArray(x, &data);
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
PETScVectorElement PETScVector::operator() (const uint i)
{
  PETScVectorElement index(i, *this);
  return index;
}
//-----------------------------------------------------------------------------
real PETScVector::operator() (const uint i) const
{
  return get(i);
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
real PETScVector::norm(const NormType type) const
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
real PETScVector::max() const
{
  dolfin_assert(x);

  int position = 0;
  real value = 0.0;
  
  VecMax(x, &position, &value);

  return value;
}
//-----------------------------------------------------------------------------
real PETScVector::min() const
{
  dolfin_assert(x);

  int  position = 0;
  real value   = 0.0;
  
  VecMin(x, &position, &value);

  return value;
}
//-----------------------------------------------------------------------------
void PETScVector::disp() const
{
  // FIXME: Maybe this could be an option?
  //VecView(x, PETSC_VIEWER_STDOUT_SELF);
 
  const uint M = size();
  cout << "[ ";
  for (uint i = 0; i < M; i++)
    cout << get(i) << " ";
  cout << "]" << endl;
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
  VecScatterBegin(x1.vec(), x2.vec(), INSERT_VALUES, SCATTER_FORWARD,
		  x1sc);
  VecScatterEnd(x1.vec(), x2.vec(), INSERT_VALUES, SCATTER_FORWARD,
		x1sc);
}
//-----------------------------------------------------------------------------
void PETScVector::scatter(PETScVector& x1, PETScVector& x2,
			   VecScatter& x1sc)
{
  VecScatterBegin(x2.vec(), x1.vec(), INSERT_VALUES, SCATTER_REVERSE,
		  x1sc);
  VecScatterEnd(x2.vec(), x1.vec(), INSERT_VALUES, SCATTER_REVERSE,
		x1sc);
}
//-----------------------------------------------------------------------------
VecScatter* PETScVector::createScatterer(PETScVector& x1, PETScVector& x2,
					  const int offset, const int size)
{
  VecScatter* sc = new VecScatter;
  IS* is = new IS;

  ISCreateBlock(MPI_COMM_WORLD, size, 1, &offset, is);
  VecScatterCreate(x1.vec(), PETSC_NULL, x2.vec(), *is,
		   sc);

  return sc;
}
//-----------------------------------------------------------------------------
void PETScVector::fromArray(const real u[], PETScVector& x,
			                      const uint offset, const uint size)
{
  // Workaround to interface PETScVector and arrays

  dolfin_assert(x);

  real* vals = x.array();
  for(uint i = 0; i < size; i++)
    vals[i] = u[i + offset];
  x.restore(vals);
}
//-----------------------------------------------------------------------------
void PETScVector::toArray(real y[], const PETScVector& x,
			                    const uint offset, const uint s)
{
  // Workaround to interface PETScVector and arrays

  const real* vals = x.array();
  for(uint i = 0; i < s; i++)
    y[offset + i] = vals[i];
  x.restore(vals);
}
//-----------------------------------------------------------------------------
void PETScVector::copy(const PETScVector& y, const uint off1, const uint off2, 
                       const uint len)
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
void PETScVector::copy(const uBlasVector& y, const uint off1, const uint off2, 
                       const uint len)
{
  // FIXME: Verify if there's a more efficient implementation

  real* vals = array();
  for(uint i = 0; i < len; i++)
    vals[i + off1] = y[i + off2];
  restore(vals);
}
//-----------------------------------------------------------------------------
// PETScVectorElement
//-----------------------------------------------------------------------------
PETScVectorElement::PETScVectorElement(const uint i, PETScVector& x) : i(i), x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVectorElement::PETScVectorElement(const PETScVectorElement& e) : i(i), x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVectorElement::operator real() const
{
  return x.get(i);
}
//-----------------------------------------------------------------------------
const PETScVectorElement& PETScVectorElement::operator=(const PETScVectorElement& e)
{
  x.set(i, e.x.get(i));

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVectorElement& PETScVectorElement::operator=(const real a)
{
  x.set(i, a);

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVectorElement& PETScVectorElement::operator+=(const real a)
{
  x.add(i, a);

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVectorElement& PETScVectorElement::operator-=(const real a)
{
  x.add(i, -a);

  return *this;
}
//-----------------------------------------------------------------------------
const PETScVectorElement& PETScVectorElement::operator*=(const real a)
{
  const real val = x.get(i) * a;
  x.set(i, val);

  return *this;
}
//-----------------------------------------------------------------------------

#endif
