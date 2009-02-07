// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2008-04-21
// Last changed:  2008-12-29

#ifdef HAS_TRILINOS

#include <cmath>
#include <cstring>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "EpetraVector.h"
#include "uBLASVector.h"
#include "PETScVector.h"
#include "EpetraFactory.h"


#include <Epetra_FEVector.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_SerialComm.h>

// FIXME: A cleanup is needed with respect to correct use of parallell vectors. 
//        This depends on decisions w.r.t. dofmaps etc in dolfin.

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraVector::EpetraVector():
    Variable("x", "a sparse vector"),
    x(static_cast<Epetra_FEVector*>(0))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(uint N):
    Variable("x", "a sparse vector"), 
    x(static_cast<Epetra_FEVector*>(0))
{
  // Create Epetra vector
  resize(N);
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(boost::shared_ptr<Epetra_FEVector> x):
    Variable("x", "a vector"),
    x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const Epetra_Map& map):
    Variable("x", "a vector"),
    x(static_cast<Epetra_FEVector*>(0))
{
  error("Not implemented yet.");
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const EpetraVector& v):
    Variable("x", "a vector"),
    x(static_cast<Epetra_FEVector*>(0))
{
  *this = v;
}
//-----------------------------------------------------------------------------
EpetraVector::~EpetraVector()
{
//  Do nothing
}
//-----------------------------------------------------------------------------
void EpetraVector::resize(uint N)
{
  if (x && this->size() == N) 
    return;

  if (!x.unique())
    error("Cannot resize EpetraVector. More than one object points to the underlying Epetra object.");

  EpetraFactory& f = dynamic_cast<EpetraFactory&>(factory());
  Epetra_SerialComm Comm = f.getSerialComm();
  Epetra_Map map(N, N, 0, Comm);

  boost::shared_ptr<Epetra_FEVector> _x(new Epetra_FEVector(map));
  x = _x; 
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraVector::copy() const
{
  dolfin_assert(x);
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
  dolfin_assert(x);
  int err = x->PutScalar(0.0);
  if (err!= 0) 
    error("EpetraVector::zero: Did not manage to perform Epetra_Vector::PutScalar."); 
}
//-----------------------------------------------------------------------------
void EpetraVector::apply()
{
  dolfin_assert(x);
  int err = x->GlobalAssemble();
  if (err!= 0) 
    error("EpetraVector::apply: Did not manage to perform Epetra_Vector::GlobalAssemble."); 

  // TODO: Use this? Relates to sparsity pattern, dofmap and reassembly!  
  //x->OptimizeStorage(); 
}
//-----------------------------------------------------------------------------
void EpetraVector::disp(uint precision) const
{
  dolfin_assert(x);

  // TODO: Make this use the dolfin::cout, doesn't compile for some reason.
  x->Print(std::cout); 
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
void EpetraVector::get(double* values) const
{
  dolfin_assert(x);

  // TODO: Are these global or local values?
  int err = x->ExtractCopy(values, 0); 
  if (err!= 0) 
    error("EpetraVector::get: Did not manage to perform Epetra_Vector::ExtractCopy."); 
}
//-----------------------------------------------------------------------------
void EpetraVector::set(double* values)
{
  dolfin_assert(x);

  int err = x->GlobalAssemble();
  if (err!= 0) 
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::GlobalAssemble."); 

  int* rows = new int[size()]; 
  for (uint i=0; i<size(); i++){
    rows[i] = i;
  }

  err = x->ReplaceGlobalValues(size(), reinterpret_cast<const int*>(rows), values);
  if (err!= 0) 
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::ReplaceGlobalValues."); 

  delete [] rows;

  /* OLD CODE, should be faster but is not bullet proof ... 
  dolfin_assert(x);
  double *data = 0;
  x->ExtractView(&data, 0);
  memcpy(data, values, size()*sizeof(double));
  */
}
//-----------------------------------------------------------------------------
void EpetraVector::add(double* values)
{
  dolfin_assert(x);

  // TODO: Use an Epetra function for this
  double *data = 0; 
  int err = x->ExtractView(&data, 0);
  if (err!= 0) 
    error("EpetraVector::add: Did not manage to perform Epetra_Vector::ExtractView."); 

  for(uint i=0; i<size(); i++)
    data[i] += values[i]; 
}
//-----------------------------------------------------------------------------
void EpetraVector::get(double* block, uint m, const uint* rows) const
{
  dolfin_assert(x);

  // TODO: use Epetra_Vector function for efficiency and parallel handling
  for (uint i=0; i<m; i++)
    block[i] = (*x)[0][rows[i]];
}
//-----------------------------------------------------------------------------
void EpetraVector::set(const double* block, uint m, const uint* rows)
{
  dolfin_assert(x);
  int err = x->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows), block);
  if (err!= 0) 
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::ReplaceGlobalValues."); 
}
//-----------------------------------------------------------------------------
void EpetraVector::add(const double* block, uint m, const uint* rows)
{
  dolfin_assert(x);
  int err = x->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows), block);
  if (err!= 0) 
    error("EpetraVector::add : Did not manage to perform Epetra_Vector::SumIntoGlobalValues."); 
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Epetra_FEVector> EpetraVector::vec() const
{
  dolfin_assert(x);
  return x;
}
//-----------------------------------------------------------------------------
double EpetraVector::inner(const GenericVector& y) const
{
  dolfin_assert(x);

  const EpetraVector& v = y.down_cast<EpetraVector>();
  if (!v.x) 
    error("Given vector is not initialized.");

  double a;
  int err = x->Dot(*(v.x), &a); 
  if (err!= 0) 
    error("EpetraVector::inner: Did not manage to perform Epetra_Vector::Dot."); 
  return a;
}
//-----------------------------------------------------------------------------
void EpetraVector::axpy(double a, const GenericVector& y) 
{
  dolfin_assert(x);

  const EpetraVector& v = y.down_cast<EpetraVector>();
  if (!v.x) 
    error("Given vector is not initialized.");

  if (size() != v.size())
    error("The vectors must be of the same size.");  

  int err = x->Update(a, *(v.vec()), 1.0); 
  if (err!= 0) 
    error("EpetraVector::axpy: Did not manage to perform Epetra_Vector::Update."); 
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
const EpetraVector& EpetraVector::operator= (double a)
{
  dolfin_assert(x);

  x->PutScalar(a);
  return *this; 
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const EpetraVector& v)
{
  dolfin_assert(v.x);

  // TODO: Check for self-assignment

  if (!x) 
  {
    if (!x.unique())
      error("More than one object points to the underlying Epetra object.");
    boost::shared_ptr<Epetra_FEVector> _x(new Epetra_FEVector(*(v.vec())));
    x =_x; 
  }
  else
    *x = *(v.vec()); 

  return *this; 
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator+= (const GenericVector& y)
{
  dolfin_assert(x);
  axpy(1.0, y); 
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator-= (const GenericVector& y)
{
  dolfin_assert(x);
  axpy(-1.0, y); 
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (double a)
{
  dolfin_assert(x);
  int err = x->Scale(a);
  if (err!= 0) 
    error("EpetraVector::operator*=: Did not manage to perform Epetra_Vector::Scale."); 
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator/= (double a)
{
  *this *= 1.0/a;
  return *this;
}
//-----------------------------------------------------------------------------
double EpetraVector::norm(NormType type) const
{
  dolfin_assert(x);

  double value = 0.0;
  int err=0; 
  switch (type) 
  {
    case l1:
      err = x->Norm1(&value);
      break;
    case l2:
      err = x->Norm2(&value);
      break;
    default:
      err = x->NormInf(&value);
  }
  if (err!= 0) 
    error("EpetraVector::norm: Did not manage to compute the norm."); 
  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::min() const
{
  dolfin_assert(x);

  double value = 0.0;
  int err = x->MinValue(&value);
  if (err!= 0) 
    error("EpetraVector::min: Did not manage to perform Epetra_Vector::MinValue."); 
  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::max() const
{
  dolfin_assert(x);

  double value = 0.0;
  int err = x->MaxValue(&value);
  if (err!= 0) 
    error("EpetraVector::min: Did not manage to perform Epetra_Vector::MinValue."); 
  return value;
}
//-----------------------------------------------------------------------------

#endif
