// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21
// Last changed:  2008-04-23

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
EpetraVector::EpetraVector():
    Variable("x", "a sparse vector"),
    x(0),
    _copy(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(uint N):
    Variable("x", "a sparse vector"), 
    x(0),
    _copy(false)
{
  // Create Epetra vector
  init(N);
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(Epetra_FEVector* x):
    Variable("x", "a vector"),
    x(x),
    _copy(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const Epetra_Map& map):
    Variable("x", "a vector"),
    x(0),
    _copy(false)
{
  error("Not implemented yet");
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
  if (x and this->size() == N) 
  {
    //clear();
    return;
  }

  EpetraFactory& f = dynamic_cast<EpetraFactory&>(factory());
  Epetra_SerialComm Comm = f.getSerialComm();
  Epetra_Map map(N, N, 0, Comm);

  x = new Epetra_FEVector(map); 
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraVector::create() const
{
  return new EpetraVector();
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraVector::copy() const
{
  EpetraVector* v = new EpetraVector(*this); 
  return v;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraVector::size() const
{
  dolfin_assert(x);
  return x->GlobalLength();
}
//-----------------------------------------------------------------------------
void EpetraVector::zero()
{
  dolfin_assert(x);
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
  dolfin_assert(x); 
  x->Print(std::cout); 
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
  if (!x) {  
    std::cout <<" x does not exist"<<std::endl; 
  }
  x->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows), block);
}
//-----------------------------------------------------------------------------
Epetra_FEVector& EpetraVector::vec() const
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

  x->Update(1.0,  v.vec(), a); 
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraVector::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const GenericVector& v)
{
  *this = v.down_cast<EpetraVector>();
  return *this; 
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const EpetraVector& v)
{
  dolfin_assert(v.x);
  if (!x) { 
    x = new Epetra_FEVector(v.vec()); 
  } else {
    *x = v.vec(); 
  }
  return *this; 
}

//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator+= (const GenericVector& x)
{
  this->axpy(1.0, x); 
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator-= (const GenericVector& x)
{
  this->axpy(-1.0, x); 
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (real a)
{
  dolfin_assert(x);
  x->Scale(a);
  return *this;
}
//-----------------------------------------------------------------------------


#endif
