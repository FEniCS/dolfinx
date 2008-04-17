// Copyright (C) 2008 Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-01-24
// Last changed: 2008-01-25

#ifdef HAS_TRILINOS

#include <cmath>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "EpetraVector.h"
#include "uBlasVector.h"
#include "PETScVector.h"
#include "EpetraFactory.h"
//#include <dolfin/MPI.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraVector::EpetraVector()
  : GenericVector(), 
    Variable("x", "a sparse vector"),
    x(0), _copy(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(uint N)
  : GenericVector(), 
    Variable("x", "a sparse vector"), 
    x(0), _copy(false)
{
  // Create Epetra vector
  init(N);
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(Epetra_FEVector* x)
  : GenericVector(),
    Variable("x", "a vector"),
    x(x), _copy(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const EpetraVector& v)
  : GenericVector(), 
    Variable("x", "a vector"),
    x(0), _copy(false)
{
  *this = v;
}
//-----------------------------------------------------------------------------
EpetraVector::~EpetraVector()
{
  if (x && !_copy) delete x;
}
//-----------------------------------------------------------------------------
void EpetraVector::init(uint N)
{
  if( this->size() == N)
  {
    //clear();
    return;
  }

  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraVector::create() const
{
  return new EpetraVector();
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraVector::copy() const
{
  // Not yet implemented
  error("Not yet implemented.");

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraVector::size() const
{
  return x->GlobalLength();
}
//-----------------------------------------------------------------------------
void EpetraVector::zero()
{
  x->PutScalar(0.0);
}
//-----------------------------------------------------------------------------
void EpetraVector::apply()
{
  x->GlobalAssemble();
  //x->OptimizeStorage(); // TODO: test this
}
//-----------------------------------------------------------------------------
void EpetraVector::disp(uint precision) const
{
  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const EpetraVector& x)
{
  // Check if matrix has been defined
  if ( !x.x )
  {
    stream << "[ Epetra vector (empty) ]";
    return stream;
  }
  stream << "[ Epetra vector of size " << x.size() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
void EpetraVector::get(real* values) const
{
  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraVector::set(real* values)
{
  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraVector::add(real* values)
{
  // Not yet implemented
  error("Not yet implemented.");
}
//-----------------------------------------------------------------------------
void EpetraVector::get(real* block, uint m, const uint* rows) const
{
  // TODO: use Epetra_Vector function for efficiency and parallell handling
  for (uint i=0; i<m; i++)
    block[i] = (*x)[0][rows[i]];
}
//-----------------------------------------------------------------------------
void EpetraVector::set(const real* block, uint m, const uint* rows)
{
  x->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows), block);
}
//-----------------------------------------------------------------------------
void EpetraVector::add(const real* block, uint m, const uint* rows)
{
  x->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows), block);
}
//-----------------------------------------------------------------------------
Epetra_MultiVector& EpetraVector::vec() const
{
  return *x;
}
//-----------------------------------------------------------------------------
real EpetraVector::inner(const GenericVector& y) const
{
  dolfin_assert(x);

  const EpetraVector& v = y.down_cast<EpetraVector>();
  dolfin_assert(v.x);

  real a;
  this->x->Dot(*(v.x),&a); 
  return a;
}
//-----------------------------------------------------------------------------
void EpetraVector::axpy(real a, const GenericVector& y) 
{
  dolfin_assert(x);

  const EpetraVector& v = y.down_cast<EpetraVector>();
  dolfin_assert(v.x);

  if (size() != v.size())
    error("The vectors must be of the same size.");  

  error("Not yet implemented"); 

}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraVector::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------


#endif
