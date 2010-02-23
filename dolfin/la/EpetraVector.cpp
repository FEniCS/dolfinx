// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008-2010.
//
// First added:  2008-04-21
// Last changed: 2009-08-22

#ifdef HAS_TRILINOS

#include <cmath>
#include <cstring>
#include <Epetra_FEVector.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <dolfin/main/MPI.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "uBLASVector.h"
#include "PETScVector.h"
#include "EpetraFactory.h"
#include "EpetraVector.h"

// FIXME: A review is needed with respect to correct use of parallel vectors.
//        It would be useful to store the Epetra map.

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(std::string type) : type(type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(uint N, std::string type) : type(type)
{
  // Create Epetra vector
  resize(N);
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(boost::shared_ptr<Epetra_FEVector> x) : x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const Epetra_Map& map)
{
  x.reset(new Epetra_FEVector(map));
}
//-----------------------------------------------------------------------------
EpetraVector::EpetraVector(const EpetraVector& v)
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

  if (x && !x.unique())
      error("Cannot resize EpetraVector. More than one object points to the underlying Epetra object.");

  // Get local range
  const std::pair<uint, uint> range = MPI::local_range(N);
  const uint n = range.second - range.first;

  if (N == n || type == "local")
  {
    EpetraFactory& f = EpetraFactory::instance();
    Epetra_SerialComm serial_comm = f.get_serial_comm();
    Epetra_Map map(N, N, 0, serial_comm);
    x.reset(new Epetra_FEVector(map));
  }
  else
  {
    EpetraFactory& f = EpetraFactory::instance();
    Epetra_MpiComm mpi_comm = f.get_mpi_comm();
    Epetra_Map map(N, n, 0, mpi_comm);
    x.reset(new Epetra_FEVector(map));
  }
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraVector::copy() const
{
  assert(x);
  EpetraVector* v = new EpetraVector(*this);
  return v;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraVector::size() const
{
  return x ? x->GlobalLength(): 0;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> EpetraVector::local_range() const
{
  // FIXME: Epetra does not provide a function for this since it does not
  //        require contiguous ranges. Can this function be removed
  //        from the generic interface and just use 'local_size'?

  const std::pair<uint, uint> range = MPI::local_range(size());
  const int n = range.second - range.first;
  if (x->MyLength() != n)
    error("Discrepency in local vector length for EpetraVector.");

  return std::make_pair<uint, uint>(range.first, range.second);
}
//-----------------------------------------------------------------------------
void EpetraVector::zero()
{
  assert(x);
  int err = x->PutScalar(0.0);
  if (err != 0)
    error("EpetraVector::zero: Did not manage to perform Epetra_Vector::PutScalar.");
}
//-----------------------------------------------------------------------------
void EpetraVector::apply()
{
  assert(x);
  int err = x->GlobalAssemble();
  if (err != 0)
    error("EpetraVector::apply: Did not manage to perform Epetra_Vector::GlobalAssemble.");

  // TODO: Use this? Relates to sparsity pattern, dofmap and reassembly!
  //x->OptimizeStorage();
}
//-----------------------------------------------------------------------------
std::string EpetraVector::str(bool verbose) const
{
  assert(x);

  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for EpetraVector not implemented, calling Epetra Print directly.");

    x->Print(std::cout);
  }
  else
    s << "<EpetraVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void EpetraVector::get_local(double* values) const
{
  assert(x);
  int err = x->ExtractCopy(values, 0);
  if (err!= 0)
    error("EpetraVector::get: Did not manage to perform Epetra_Vector::ExtractCopy.");
}
//-----------------------------------------------------------------------------
void EpetraVector::set_local(const double* values)
{
  assert(x);
  const std::pair<uint, uint> range = local_range();
  const uint n0 = range.first;
  const uint n1 = range.second;
  const uint N = n1 - n0;

  // FIXME: The user should call apply()
  int err = x->GlobalAssemble();

  if (err != 0)
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::GlobalAssemble.");

  std::vector<int> rows(N);
  for (uint i = 0; i < N; i++)
    rows[i] = i + n0;
  err = x->ReplaceGlobalValues(N, &rows[0], values);

  if (err!= 0)
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::ReplaceGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::add_local(const double* values)
{
  assert(x);

  const std::pair<uint, uint> range = local_range();
  const uint n0 = range.first;
  const uint n1 = range.second;
  const uint N = n1 - n0;

  std::vector<int> rows(N);
  for (uint i = 0; i < N; i++)
    rows[i] = i + n0;
  int err = x->SumIntoGlobalValues(N, &rows[0], values);
  if (err!= 0)
    error("EpetraVector::add_local: Did not manage to perform Epetra_Vector::SumIntoGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::get(double* block, uint m, const uint* rows) const
{
  assert(x);

  // Trilinos doesn't appear to provide any special functions for getting values.
  for (uint i = 0; i < m; i++)
    block[i] = (*x)[0][rows[i]];
}
//-----------------------------------------------------------------------------
void EpetraVector::set(const double* block, uint m, const uint* rows)
{
  assert(x);
  int err = x->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows), block);
  if (err!= 0)
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::ReplaceGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::add(const double* block, uint m, const uint* rows)
{
  assert(x);
  int err = x->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows), block);
  if (err!= 0)
    error("EpetraVector::add: Did not manage to perform Epetra_Vector::SumIntoGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::get_local(double* block, uint m, const uint* rows) const
{
  error("EpetraVector::get_local not working.");

  // FIXME: Use ExtractCopy for this function?
  //int err = x->ExtractCopy(values, 0);

  for (uint i = 0; i < m; ++i)
    //block[i] = (*x)[0][rows[i]];
    block[i] = 0.0;
}
//-----------------------------------------------------------------------------
void EpetraVector::gather(GenericVector& y,
                          const std::vector<dolfin::uint>& indices) const
{
  error("EpetraVector::gatther not working.");

  assert(x);

  // Down cast to a EpetraVector and resize
  EpetraVector& _y = y.down_cast<EpetraVector>();
  _y.resize(indices.size());

  // Check that y is a local vector (check communicator)


  // Get data from x and insert into y
  for (uint i = 0; i < indices.size(); ++i)
  {
    cout << "Testing in gather " << i << "  " << indices[i] << "  " << (*x)[0][indices[i]] << endl;
    //(*_y.vec())[0][i] = 0.0;
    (*_y.vec())[0][i] = (*x)[0][indices[i]];
  }
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Epetra_FEVector> EpetraVector::vec() const
{
  assert(x);
  return x;
}
//-----------------------------------------------------------------------------
double EpetraVector::inner(const GenericVector& y) const
{
  assert(x);

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
  assert(x);

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
  assert(x);

  x->PutScalar(a);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator= (const EpetraVector& v)
{
  assert(v.x);

  // TODO: Check for self-assignment
  if (!x)
    x.reset(new Epetra_FEVector(*(v.vec())));
  else
    *x = *(v.vec());

  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator+= (const GenericVector& y)
{
  assert(x);
  axpy(1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator-= (const GenericVector& y)
{
  assert(x);
  axpy(-1.0, y);
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (double a)
{
  assert(x);
  int err = x->Scale(a);
  if (err!= 0)
    error("EpetraVector::operator*=: Did not manage to perform Epetra_Vector::Scale.");
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator*= (const GenericVector& y)
{
  assert(x);

  const EpetraVector& v = y.down_cast<EpetraVector>();
  if (!v.x)
    error("Given vector is not initialized.");

  if (size() != v.size())
    error("The vectors must be of the same size.");

  int err = x->Multiply(1.0,*x,*v.x,0.0);
  if (err!= 0)
    error("EpetraVector::operator*=: Did not manage to perform Epetra_Vector::Multiply.");
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraVector& EpetraVector::operator/=(double a)
{
  *this *= 1.0/a;
  return *this;
}
//-----------------------------------------------------------------------------
double EpetraVector::norm(std::string norm_type) const
{
  assert(x);

  double value = 0.0;
  int err = 0;
  if (norm_type == "l1")
    err = x->Norm1(&value);
  else if (norm_type == "l2")
    err = x->Norm2(&value);
  else
    err = x->NormInf(&value);

  if (err != 0)
    error("EpetraVector::norm: Did not manage to compute the norm.");
  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::min() const
{
  assert(x);

  double value = 0.0;
  int err = x->MinValue(&value);
  if (err!= 0)
    error("EpetraVector::min: Did not manage to perform Epetra_Vector::MinValue.");
  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::max() const
{
  assert(x);

  double value = 0.0;
  int err = x->MaxValue(&value);
  if (err != 0)
    error("EpetraVector::min: Did not manage to perform Epetra_Vector::MinValue.");
  return value;
}
//-----------------------------------------------------------------------------
double EpetraVector::sum() const
{
  assert(x);

  const std::pair<uint, uint> range = local_range();
  const uint n0 = range.first;
  const uint n1 = range.second;
  const uint N = n1 - n0;

  // Get local values
  std::vector<double> x_local(N);
  get_local(&x_local[0]);

  // Compute local sum
  double local_sum = 0.0;
  for (uint i = 0; i < N; ++i)
    local_sum += x_local[0];

  // Compute global sum
  double global_sum = 0.0;
  x->Comm().SumAll(&local_sum, &global_sum, 1);

  return global_sum;
}
//-----------------------------------------------------------------------------
#endif
