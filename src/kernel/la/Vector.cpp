// Copyright (C) 2004-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2004
// Last changed: 2005-11-01

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/Vector.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
Vector::Vector() : x(0), copy(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
Vector::Vector(uint size) : x(0), copy(false)
{
  if(size < 0)
    dolfin_error("Size of vector must be non-negative.");

  // Initialize PETSc
  PETScManager::init();

  // Create PETSc vector
  init(size);
}
//-----------------------------------------------------------------------------
Vector::Vector(Vec x) : x(x), copy(true)
{
  // Initialize PETSc 
  PETScManager::init();
}
//-----------------------------------------------------------------------------
Vector::Vector(const Vector& v) : x(0), copy(false)
{
  // Initialize PETSc 
  PETScManager::init();

  *this = v;
}
//-----------------------------------------------------------------------------
Vector::~Vector()
{
  clear();
}
//-----------------------------------------------------------------------------
void Vector::init(uint size)
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
void Vector::axpy(const real a, const Vector& x) const
{
  VecAXPY(this->x, a, x.vec());
}
//-----------------------------------------------------------------------------
void Vector::div(const Vector& x, Vector& y) const
{
  VecPointwiseDivide(this->x, y.vec(), x.vec());
  y.apply();
}
//-----------------------------------------------------------------------------
void Vector::mult(const Vector& x, Vector& y) const
{
  VecPointwiseMult(this->x, y.vec(), x.vec());
  y.apply();
}
//-----------------------------------------------------------------------------
void Vector::add(const real block[], const int cols[], int n)
{
  VecSetValues(x, n, cols, block, ADD_VALUES); 
}
//-----------------------------------------------------------------------------
void Vector::apply()
{
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
void Vector::clear()
{
  if ( x && !copy )
  {
    VecDestroy(x);
  }

  x = 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Vector::size() const
{
  int n = 0;
  VecGetSize(x, &n);

  return static_cast<uint>(n);
}
//-----------------------------------------------------------------------------
Vec Vector::vec()
{
  return x;
}
//-----------------------------------------------------------------------------
const Vec Vector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
real* Vector::array()
{
  dolfin_assert(x);

  real* data = 0;
  VecGetArray(x, &data);
  dolfin_assert(data);

  return data;
}
//-----------------------------------------------------------------------------
const real* Vector::array() const
{
  dolfin_assert(x);

  real* data = 0;
  VecGetArray(x, &data);

  return data;
}
//-----------------------------------------------------------------------------
void Vector::restore(real data[])
{
  dolfin_assert(x);

  VecRestoreArray(x, &data);
}
//-----------------------------------------------------------------------------
void Vector::restore(const real data[]) const
{
  dolfin_assert(x);

  // Cast away the constness and trust PETSc to do the right thing
  real* tmp = const_cast<real *>(data);
  VecRestoreArray(x, &tmp);
}
//-----------------------------------------------------------------------------
VectorElement Vector::operator() (uint i)
{
  VectorElement index(i, *this);
  return index;
}
//-----------------------------------------------------------------------------
real Vector::operator() (uint i) const
{
  return getval(i);
}
//-----------------------------------------------------------------------------
const Vector& Vector::operator= (const Vector& x)
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
const Vector& Vector::operator= (real a)
{
  dolfin_assert(x);
  VecSet(x, a);

  return *this;
}
//-----------------------------------------------------------------------------
const Vector& Vector::operator+= (const Vector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  const real a = 1.0;
  VecAXPY(this->x, a, x.x);

  return *this;
}
//-----------------------------------------------------------------------------
const Vector& Vector::operator-= (const Vector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  const real a = -1.0;
  VecAXPY(this->x, a, x.x);

  return *this;
}
//-----------------------------------------------------------------------------
const Vector& Vector::operator*= (real a)
{
  dolfin_assert(x);
  VecScale(x, a);
  
  return *this;
}
//-----------------------------------------------------------------------------
const Vector& Vector::operator/= (real a)
{
  dolfin_assert(x);
  dolfin_assert(a != 0.0);

  const real b = 1.0 / a;
  VecScale(x, b);
  
  return *this;
}
//-----------------------------------------------------------------------------
real Vector::operator*(const Vector& x)
{
  dolfin_assert(x.x);
  dolfin_assert(this->x);

  real a;
  VecDot(x.x, this->x, &a);

  return a;
}
//-----------------------------------------------------------------------------
real Vector::norm(NormType type) const
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
real Vector::sum() const
{
  dolfin_assert(x);

  real value = 0.0;
  
  VecSum(x, &value);
  
  return value;
}
//-----------------------------------------------------------------------------
real Vector::max() const
{
  dolfin_assert(x);

  int position = 0;
  real value = 0.0;
  
  VecMax(x, &position, &value);

  return value;
}
//-----------------------------------------------------------------------------
real Vector::min() const
{
  dolfin_assert(x);

  int  position = 0;
  real value   = 0.0;
  
  VecMin(x, &position, &value);

  return value;
}
//-----------------------------------------------------------------------------
void Vector::disp() const
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
LogStream& dolfin::operator<< (LogStream& stream, const Vector& x)
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
real Vector::getval(uint i) const
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
void Vector::setval(uint i, const real a)
{
  dolfin_assert(x);

  VecSetValue(x, static_cast<int>(i), a, INSERT_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
void Vector::addval(uint i, const real a)
{
  dolfin_assert(x);

  VecSetValue(x, static_cast<int>(i), a, ADD_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
// VectorElement
//-----------------------------------------------------------------------------
VectorElement::VectorElement(uint i, Vector& x) : i(i), x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
VectorElement::VectorElement(const VectorElement& e) : i(i), x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
VectorElement::operator real() const
{
  return x.getval(i);
}
//-----------------------------------------------------------------------------
const VectorElement& VectorElement::operator=(const VectorElement& e)
{
  x.setval(i, e.x.getval(i));

  return *this;
}
//-----------------------------------------------------------------------------
const VectorElement& VectorElement::operator=(const real a)
{
  x.setval(i, a);

  return *this;
}
//-----------------------------------------------------------------------------
const VectorElement& VectorElement::operator+=(const real a)
{
  x.addval(i, a);

  return *this;
}
//-----------------------------------------------------------------------------
const VectorElement& VectorElement::operator-=(const real a)
{
  x.addval(i, -a);

  return *this;
}
//-----------------------------------------------------------------------------
const VectorElement& VectorElement::operator*=(const real a)
{
  const real val = x.getval(i) * a;
  x.setval(i, val);

  return *this;
}
//-----------------------------------------------------------------------------
