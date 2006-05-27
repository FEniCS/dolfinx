// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2004
// Last changed: 2006-05-15

// FIXME: Insert dolfin_assert() where appropriate

#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/SparseVector.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
SparseVector::SparseVector()
  : GenericVector(), 
    Variable("x", "a sparse vector"),
    x(0), copy(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
SparseVector::SparseVector(uint size)
  : GenericVector(), 
    Variable("x", "a sparse vector"), 
    x(0), copy(false)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc vector
  init(size);
}
//-----------------------------------------------------------------------------
SparseVector::SparseVector(Vec x)
  : GenericVector(),
    Variable("x", "a vector"),
    x(x), copy(true)
{
  // Initialize PETSc 
  PETScManager::init();
}
//-----------------------------------------------------------------------------
SparseVector::SparseVector(const SparseVector& v)
  : GenericVector(), 
    Variable("x", "a vector"),
    x(0), copy(false)
{
  // Initialize PETSc 
  PETScManager::init();

  *this = v;
}
//-----------------------------------------------------------------------------
SparseVector::~SparseVector()
{
  clear();
}
//-----------------------------------------------------------------------------
void SparseVector::init(uint size)
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

    if (n == size)
    {
      return;      
    }
  }
  else
  {
    clear();
  }

  // Create vector
  VecCreate(PETSC_COMM_SELF, &x);
  VecSetSizes(x, PETSC_DECIDE, size);
  VecSetFromOptions(x);

  // Set all entries to zero
  PetscScalar a = 0.0;
  VecSet(x, a);
}
//-----------------------------------------------------------------------------
void SparseVector::axpy(const real a, const SparseVector& x) const
{
  VecAXPY(this->x, a, x.vec());
}
//-----------------------------------------------------------------------------
void SparseVector::div(const SparseVector& x)
{
  VecPointwiseDivide(this->x, this->x, x.vec());
  apply();
}
//-----------------------------------------------------------------------------
void SparseVector::mult(const SparseVector& x)
{
  VecPointwiseMult(this->x, this->x, x.vec());
  apply();
}
//-----------------------------------------------------------------------------
void SparseVector::set(const real block[], const int cols[], int n)
{
  VecSetValues(x, n, cols, block, INSERT_VALUES); 
}
//-----------------------------------------------------------------------------
void SparseVector::add(const real block[], const int cols[], int n)
{
  dolfin_assert(x);
  VecSetValues(x, n, cols, block, ADD_VALUES); 
}
//-----------------------------------------------------------------------------
void SparseVector::get(real block[], const int cols[], int n) const
{
  dolfin_assert(x);
  VecGetValues(x, n, cols, block); 
}
//-----------------------------------------------------------------------------
void SparseVector::apply()
{
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
void SparseVector::zero()
{
  dolfin_assert(x);
  real a = 0.0;
  VecSet(x, a);
}
//-----------------------------------------------------------------------------
void SparseVector::clear()
{
  if ( x && !copy )
  {
    VecDestroy(x);
  }

  x = 0;
}
//-----------------------------------------------------------------------------
dolfin::uint SparseVector::size() const
{
  int n = 0;
  VecGetSize(x, &n);

  return static_cast<uint>(n);
}
//-----------------------------------------------------------------------------
Vec SparseVector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
real* SparseVector::array()
{
  dolfin_assert(x);

  real* data = 0;
  VecGetArray(x, &data);
  dolfin_assert(data);

  return data;
}
//-----------------------------------------------------------------------------
const real* SparseVector::array() const
{
  dolfin_assert(x);

  real* data = 0;
  VecGetArray(x, &data);

  return data;
}
//-----------------------------------------------------------------------------
void SparseVector::restore(real data[])
{
  dolfin_assert(x);

  VecRestoreArray(x, &data);
}
//-----------------------------------------------------------------------------
void SparseVector::restore(const real data[]) const
{
  dolfin_assert(x);

  // Cast away the constness and trust PETSc to do the right thing
  real* tmp = const_cast<real *>(data);
  VecRestoreArray(x, &tmp);
}
//-----------------------------------------------------------------------------
SparseVectorElement SparseVector::operator() (uint i)
{
  SparseVectorElement index(i, *this);
  return index;
}
//-----------------------------------------------------------------------------
real SparseVector::operator() (uint i) const
{
  return getval(i);
}
//-----------------------------------------------------------------------------
const SparseVector& SparseVector::operator= (const SparseVector& x)
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
const SparseVector& SparseVector::operator= (real a)
{
  dolfin_assert(x);
  VecSet(x, a);

  return *this;
}
//-----------------------------------------------------------------------------
const SparseVector& SparseVector::operator+= (const SparseVector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  const real a = 1.0;
  VecAXPY(this->x, a, x.x);

  return *this;
}
//-----------------------------------------------------------------------------
const SparseVector& SparseVector::operator-= (const SparseVector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  const real a = -1.0;
  VecAXPY(this->x, a, x.x);

  return *this;
}
//-----------------------------------------------------------------------------
const SparseVector& SparseVector::operator*= (real a)
{
  dolfin_assert(x);
  VecScale(x, a);
  
  return *this;
}
//-----------------------------------------------------------------------------
const SparseVector& SparseVector::operator/= (real a)
{
  dolfin_assert(x);
  dolfin_assert(a != 0.0);

  const real b = 1.0 / a;
  VecScale(x, b);
  
  return *this;
}
//-----------------------------------------------------------------------------
real SparseVector::operator*(const SparseVector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  real a;
  VecDot(x.x, this->x, &a);

  return a;
}
//-----------------------------------------------------------------------------
real SparseVector::norm(NormType type) const
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
real SparseVector::sum() const
{
  dolfin_assert(x);

  real value = 0.0;
  
  VecSum(x, &value);
  
  return value;
}
//-----------------------------------------------------------------------------
real SparseVector::max() const
{
  dolfin_assert(x);

  int position = 0;
  real value = 0.0;
  
  VecMax(x, &position, &value);

  return value;
}
//-----------------------------------------------------------------------------
real SparseVector::min() const
{
  dolfin_assert(x);

  int  position = 0;
  real value   = 0.0;
  
  VecMin(x, &position, &value);

  return value;
}
//-----------------------------------------------------------------------------
void SparseVector::disp() const
{
  // FIXME: Maybe this could be an option?
  //VecView(x, PETSC_VIEWER_STDOUT_SELF);
 
  const uint M = size();
  cout << "[ ";
  for (uint i = 0; i < M; i++)
    cout << getval(i) << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const SparseVector& x)
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
real SparseVector::getval(uint i) const
{
  dolfin_assert(x);

  // Assumes uniprocessor case

  real val = 0.0;

  PetscScalar *array = 0;
  VecGetArray(x, &array);
  val = array[i];
  VecRestoreArray(x, &array);

  return val;
}
//-----------------------------------------------------------------------------
void SparseVector::setval(uint i, const real a)
{
  dolfin_assert(x);

  VecSetValue(x, static_cast<int>(i), a, INSERT_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
void SparseVector::addval(uint i, const real a)
{
  dolfin_assert(x);

  VecSetValue(x, static_cast<int>(i), a, ADD_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
void SparseVector::gather(SparseVector& x1, SparseVector& x2, VecScatter& x1sc)
{
  VecScatterBegin(x1.vec(), x2.vec(), INSERT_VALUES, SCATTER_FORWARD,
		  x1sc);
  VecScatterEnd(x1.vec(), x2.vec(), INSERT_VALUES, SCATTER_FORWARD,
		x1sc);
}
//-----------------------------------------------------------------------------
void SparseVector::scatter(SparseVector& x1, SparseVector& x2,
			   VecScatter& x1sc)
{
  VecScatterBegin(x2.vec(), x1.vec(), INSERT_VALUES, SCATTER_REVERSE,
		  x1sc);
  VecScatterEnd(x2.vec(), x1.vec(), INSERT_VALUES, SCATTER_REVERSE,
		x1sc);
}
//-----------------------------------------------------------------------------
VecScatter* SparseVector::createScatterer(SparseVector& x1, SparseVector& x2,
					  int offset, int size)
{
  VecScatter* sc = new VecScatter;
  IS* is = new IS;

  ISCreateBlock(MPI_COMM_WORLD, size, 1, &offset, is);
  VecScatterCreate(x1.vec(), PETSC_NULL, x2.vec(), *is,
		   sc);

  return sc;
}
//-----------------------------------------------------------------------------
void SparseVector::fromArray(const real u[], SparseVector& x,
			     unsigned int offset,
			     unsigned int size)
{
  // Workaround to interface SparseVector and arrays

  real* vals = 0;
  vals = x.array();
  for(unsigned int i = 0; i < size; i++)
  {
    vals[i] = u[i + offset];
  }
  x.restore(vals);
}
//-----------------------------------------------------------------------------
void SparseVector::toArray(real y[], SparseVector& x,
			   unsigned int offset,
			   unsigned int size)
{
  // Workaround to interface SparseVector and arrays

  real* vals = 0;
  vals = x.array();
  for(unsigned int i = 0; i < size; i++)
  {
    y[offset + i] = vals[i];
  }
  x.restore(vals);
}
//-----------------------------------------------------------------------------
// SparseVectorElement
//-----------------------------------------------------------------------------
SparseVectorElement::SparseVectorElement(uint i, SparseVector& x) : i(i), x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SparseVectorElement::SparseVectorElement(const SparseVectorElement& e) : i(i), x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SparseVectorElement::operator real() const
{
  return x.getval(i);
}
//-----------------------------------------------------------------------------
const SparseVectorElement& SparseVectorElement::operator=(const SparseVectorElement& e)
{
  x.setval(i, e.x.getval(i));

  return *this;
}
//-----------------------------------------------------------------------------
const SparseVectorElement& SparseVectorElement::operator=(const real a)
{
  x.setval(i, a);

  return *this;
}
//-----------------------------------------------------------------------------
const SparseVectorElement& SparseVectorElement::operator+=(const real a)
{
  x.addval(i, a);

  return *this;
}
//-----------------------------------------------------------------------------
const SparseVectorElement& SparseVectorElement::operator-=(const real a)
{
  x.addval(i, -a);

  return *this;
}
//-----------------------------------------------------------------------------
const SparseVectorElement& SparseVectorElement::operator*=(const real a)
{
  const real val = x.getval(i) * a;
  x.setval(i, val);

  return *this;
}
//-----------------------------------------------------------------------------

#endif
