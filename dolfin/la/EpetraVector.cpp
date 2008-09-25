// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21
// Last changed:  2008-07-04

#ifdef HAS_TRILINOS

#include <cmath>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "EpetraVector.h"
#include "uBLASVector.h"
#include "PETScVector.h"
#include "EpetraFactory.h"
//#include <dolfin/MPI.h>


#include <Epetra_FEVector.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_SerialComm.h>

// FIXME: A cleanup is needed with respect to correct use of parallell vectors. This depends on decisions w.r.t. dofmaps etc in dolfin.

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraVector::EpetraVector():
    Variable("x", "a sparse vector"),
    x(0),
    is_view(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(uint N):
    Variable("x", "a sparse vector"), 
    x(0),
    is_view(false)
{
  // Create Epetra vector
  resize(N);
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(Epetra_FEVector* x):
    Variable("x", "a vector"),
    x(x),
    is_view(true)
{
  // Do nothing // TODO: Awaiting memory ownership conventions in DOLFIN!
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const Epetra_Map& map):
    Variable("x", "a vector"),
    x(0),
    is_view(false)
{
  error("Not implemented yet");
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const EpetraVector& v):
    Variable("x", "a vector"),
    x(0),
    is_view(false)
{
  *this = v;
}
//-----------------------------------------------------------------------------
EpetraVector::~EpetraVector()
{
  if (x && !is_view) 
    delete x;
}
//-----------------------------------------------------------------------------
void EpetraVector::resize(uint N)
{
  if (x && this->size() == N) 
    return;

  EpetraFactory& f = dynamic_cast<EpetraFactory&>(factory());
  Epetra_SerialComm Comm = f.getSerialComm();
  Epetra_Map map(N, N, 0, Comm);

  x = new Epetra_FEVector(map); 
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraVector::copy() const
{
  if (!x) 
    error("Vector is not initialized.");

  EpetraVector* v = new EpetraVector(*this); 
  return v;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraVector::size() const
{
  return x ? x->GlobalLength(): 0;
}
//-----------------------------------------------------------------------------
void EpetraVector::zero()
{
  if (!x) error("Vector is not initialized.");
  x->PutScalar(0.0);
}
//-----------------------------------------------------------------------------
void EpetraVector::apply()
{
  if (!x) error("Vector is not initialized.");
  x->GlobalAssemble();
  //x->OptimizeStorage(); // TODO: Use this? Relates to sparsity pattern, dofmap and reassembly!
}
//-----------------------------------------------------------------------------
void EpetraVector::disp(uint precision) const
{
  if (!x) error("Vector is not initialized.");
  x->Print(std::cout); // TODO: Make this use the dolfin::cout, doesn't compile for some reason.
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const EpetraVector& x)
{
  // Check if matrix has been defined
  if ( x.size() == 0 )
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
  if (!x) error("Vector is not initialized.");
  x->ExtractCopy(values, 0); // TODO: Are these global or local values?
}
//-----------------------------------------------------------------------------
void EpetraVector::set(real* values)
{
  if (!x) error("Vector is not initialized.");
  double *data = 0;
  x->ExtractView(&data, 0);
  memcpy(data, values, size()*sizeof(double));
}
//-----------------------------------------------------------------------------
void EpetraVector::add(real* values)
{
  if (!x) error("Vector is not initialized.");
  double *data = 0; 
  x->ExtractView(&data, 0);
  for(uint i=0; i<size(); i++)
    data[i] += values[i]; // TODO: Use an Epetra function for this!
}
//-----------------------------------------------------------------------------
void EpetraVector::get(real* block, uint m, const uint* rows) const
{
  dolfin_assert(x);
  // TODO: use Epetra_Vector function for efficiency and parallell handling
  for (uint i=0; i<m; i++)
    block[i] = (*x)[0][rows[i]];
}
//-----------------------------------------------------------------------------
void EpetraVector::set(const real* block, uint m, const uint* rows)
{
  dolfin_assert(x);
  int err = x->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows), block);
  if (err!= 0) error("EpetraVector::set: Did not manage to set the values into the vector"); 
}
//-----------------------------------------------------------------------------
void EpetraVector::add(const real* block, uint m, const uint* rows)
{
  dolfin_assert(x);
  int err = x->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows), block);
  if (err!= 0) error("EpetraVector::add : Did not manage to add the values to the vector"); 
}
//-----------------------------------------------------------------------------
Epetra_FEVector& EpetraVector::vec() const
{
  if (!x) error("Vector is not initialized.");
  return *x;
}
//-----------------------------------------------------------------------------
real EpetraVector::inner(const GenericVector& y) const
{
  if (!x) error("Vector is not initialized.");

  const EpetraVector& v = y.down_cast<EpetraVector>();
  if (!v.x) error("Given vector is not initialized.");

  real a;
  x->Dot(*(v.x), &a); 
  return a;
}
//-----------------------------------------------------------------------------
void EpetraVector::axpy(real a, const GenericVector& y) 
{
  if (!x) error("Vector is not initialized.");

  const EpetraVector& v = y.down_cast<EpetraVector>();
  if (!v.x) error("Given vector is not initialized.");

  if (size() != v.size())
    error("The vectors must be of the same size.");  

  int err = x->Update(a, v.vec(), 1.0); 
  if (err!= 0) error("EpetraVector::axpy: Did not manage to perform Update on Epetra vector."); 
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraVector::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const GenericVector& v)
{
  *this = v.down_cast<EpetraVector>();
  return *this; 
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (real a)
{
  if (!x) error("Vector is not initialized.");
  x->PutScalar(a);
  return *this; 
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const EpetraVector& v)
{
  if (!v.x) error("Given vector is not initialized.");
  if (!x) { 
    x = new Epetra_FEVector(v.vec()); 
  } else {
    *x = v.vec(); 
  }
  return *this; 
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator+= (const GenericVector& y)
{
  if (!x) error("Vector is not initialized.");
  axpy(1.0, y); 
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator-= (const GenericVector& y)
{
  if (!x) error("Vector is not initialized.");
  axpy(-1.0, y); 
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (real a)
{
  if (!x) error("Vector is not initialized.");
  x->Scale(a);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator/= (real a)
{
  *this *= 1.0 / a;
  return *this;
}
//-----------------------------------------------------------------------------
real EpetraVector::norm(NormType type) const
{
  if (!x) error("Vector is not initialized.");
  real value = 0.0;
  switch (type) {
  case l1:
    x->Norm1(&value);
    break;
  case l2:
    x->Norm2(&value);
    break;
  default:
    x->NormInf(&value);
  }
  return value;
}
//-----------------------------------------------------------------------------
real EpetraVector::min() const
{
  if (!x) error("Vector is not initialized.");
  real value = 0.0;
  x->MinValue(&value);
  return value;
}
//-----------------------------------------------------------------------------
real EpetraVector::max() const
{
  if (!x) error("Vector is not initialized.");
  real value = 0.0;
  x->MaxValue(&value);
  return value;
}
//-----------------------------------------------------------------------------

#endif
