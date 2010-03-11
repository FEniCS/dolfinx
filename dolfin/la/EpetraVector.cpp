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
#include <numeric>

#include <Epetra_FEVector.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <Epetra_Vector.h>

#include <dolfin/common/Array.h>
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
  // Do nothing
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
  return new EpetraVector(*this);
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraVector::size() const
{
  return x ? x->GlobalLength(): 0;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> EpetraVector::local_range() const
{
  assert(x);

  if ( x->Comm().NumProc() == 1 )
    return std::make_pair<uint, uint>(0, size());
  else
  {
    assert(x->Map().LinearMap());
    const Epetra_BlockMap& map = x->Map();
    return std::make_pair<uint, uint>(map.MinMyGID(), map.MaxMyGID() + 1);
  }
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
void EpetraVector::apply(std::string mode)
{
  assert(x);
  int err = 0;
  if (mode == "add")
    err = x->GlobalAssemble(Add);
  else if (mode == "insert")
    err = x->GlobalAssemble(Insert);
  else
    error("Unknown apply mode in EpetraVector::apply");

  if (err != 0)
    error("EpetraVector::apply: Did not manage to perform Epetra_Vector::GlobalAssemble.");
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
void EpetraVector::get_local(Array<double>& values) const
{
  assert(x);
  assert( (int) values.size() >= x->MyLength());
  int err = x->ExtractCopy(values.data().get(), 0);
  if (err!= 0)
    error("EpetraVector::get: Did not manage to perform Epetra_Vector::ExtractCopy.");
}
//-----------------------------------------------------------------------------
void EpetraVector::set_local(const Array<double>& values)
{
  assert(x);
  assert((int) values.size() >= x->MyLength());
  const uint n0 = local_range().first;
  const uint n1 = local_range().second;
  const uint N = n1 - n0;

  // FIXME: Set data directly

  std::vector<int> rows(N);
  for (uint i = 0; i < N; i++)
    rows[i] = i + n0;
  int err = x->ReplaceGlobalValues(N, &rows[0], values.data().get());

  if (err!= 0)
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::ReplaceGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::add_local(const Array<double>& values)
{
  assert(x);
  assert((int) values.size() == x->MyLength());

  const uint n0 = local_range().first;
  const uint n1 = local_range().second;
  const uint N = n1 - n0;

  // FIXME: Set data directly

  std::vector<int> rows(N);
  for (uint i = 0; i < N; i++)
    rows[i] = i + n0;
  int err = x->SumIntoGlobalValues(N, &rows[0], values.data().get());
  if (err!= 0)
    error("EpetraVector::add_local: Did not manage to perform Epetra_Vector::SumIntoGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::get(double* block, uint m, const uint* rows) const
{
  // If vector is local this function will call get_local. For distributed 
  // vectors, perform first a gather into a local vector

  if (local_range().first == 0 && local_range().second == size())
    get_local(block, m, rows);
  else
  {
    EpetraVector y("local");
    std::vector<uint> indices;
    std::vector<uint> local_indices;
    indices.reserve(m);
    local_indices.reserve(m);
    for (uint i = 0; i < m; ++i)
    {
      indices.push_back(rows[i]);
      local_indices.push_back(i);
    }

    // Gather values into y
    const Array<uint> _indices(indices.size(), &indices[0]);
    gather(y, _indices);

    // Get entries of y
    y.get_local(block, m, &local_indices[0]);
  }
}
//-----------------------------------------------------------------------------
void EpetraVector::set(const double* block, uint m, const uint* rows)
{
  assert(x);
  int err = x->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows), block, 0);
  if (err != 0)
    error("EpetraVector::set: Did not manage to perform Epetra_Vector::ReplaceGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::add(const double* block, uint m, const uint* rows)
{
  assert(x);
  int err = x->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows), block, 0);
  if (err != 0)
    error("EpetraVector::add: Did not manage to perform Epetra_Vector::SumIntoGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraVector::get_local(double* block, uint m, const uint* rows) const
{
  assert(x);
  const uint n0 = local_range().first;
  for (uint i = 0; i < m; ++i)
    block[i] = (*x)[0][rows[i] - n0];
}
//-----------------------------------------------------------------------------
void EpetraVector::gather(GenericVector& y,
                          const Array<dolfin::uint>& indices) const
{
  // FIXME: This can be done better. Problem is that the GenericVector interface
  //        is PETSc-centric for the parallel case. It should be improved.

  assert(x);

  EpetraFactory& f = EpetraFactory::instance();
  //Epetra_MpiComm Comm = f.get_mpi_comm();
  Epetra_SerialComm serial_comm = f.get_serial_comm();

  // Down cast to a EpetraVector and resize
  EpetraVector& _y = y.down_cast<EpetraVector>();

  // Check that y is a local vector (check communicator)

  // Create map
  std::vector<int> _indices(indices.size());
  for (uint i = 0; i < indices.size(); ++i)
    _indices[i] = indices[i];

  //Epetra_Map source_map(-1, indices.size(), &_indices[0], 0, Comm);
  //Epetra_Map target_map(-1, indices.size(), &_indices[0], 0, Comm);
  Epetra_Map target_map(indices.size(), indices.size(), &_indices[0], 0, serial_comm);

  // FIXME: Check that the data belonging to y is not shared
  _y.reset(target_map);
  //boost::shared_ptr<Epetra_FEVector> yy(new Epetra_FEVector(target_map));
  Epetra_Import importer(_y.vec()->Map(), x->Map());
  _y.vec()->Import(*x, importer, Insert);

  //x->Print(std::cout);

  //cout << "New vector" << endl;
  //_y.vec()->Print(std::cout);
  //cout << "End New vector" << endl;
}
//-----------------------------------------------------------------------------
void EpetraVector::reset(const Epetra_Map& map)
{
  x.reset(new Epetra_FEVector(map));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Epetra_FEVector> EpetraVector::vec() const
{
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

  const EpetraVector& _y = y.down_cast<EpetraVector>();
  if (!_y.x)
    error("Given vector is not initialized.");

  if (size() != _y.size())
    error("The vectors must be of the same size.");

  int err = x->Update(a, *(_y.vec()), 1.0);

  if (err != 0)
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

  int err = x->Multiply(1.0, *x, *v.x, 0.0);
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
  Array<double> x_local(N);
  get_local(x_local);

  // Compute local sum
  //double local_sum = std::accumulate(x_local.begin(), x_local.end(), 0.0);
  double local_sum = 0.0;
  for (uint i = 0; i < N; ++i)
    local_sum += x_local[i];

  // Compute global sum
  double global_sum = 0.0;
  x->Comm().SumAll(&local_sum, &global_sum, 1);

  return global_sum;
}
//-----------------------------------------------------------------------------
double EpetraVector::sum(const Array<uint>& rows) const
{
  assert(x);

  const std::pair<uint, uint> range = local_range();
  const uint n0 = range.first;
  const uint n1 = range.second;
  const uint N = n1 - n0;

  // Get local values
  Array<double> x_local(N);
  get_local(x_local);

  // Sum on-process entries
  double local_sum = 0.0;
  for (uint i = 0; i < rows.size(); ++i)
  {
    if (rows[i] <= n0 && rows[i] < n1)
      local_sum += x_local[ rows[i] - n0 ];
  }

  // Compute global sum
  double global_sum = 0.0;
  x->Comm().SumAll(&local_sum, &global_sum, 1);

  return global_sum;
}
//-----------------------------------------------------------------------------

#endif
