// Copyright (C) 2004-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2009.
// Modified by Martin Sandve Alnes 2008
//
// First added:  2004
// Last changed: 2009-09-07

#ifdef HAS_PETSC

#include <cmath>
#include <boost/assign/list_of.hpp>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScVector.h"
#include "uBLASVector.h"
#include "PETScFactory.h"
#include <dolfin/main/MPI.h>

namespace dolfin
{
  class PETScVectorDeleter
  {
  public:
    void operator() (Vec* x)
    {
      if (x)
        VecDestroy(*x);
      delete x;
    }
  };
}

using namespace dolfin;

const std::map<std::string, NormType> PETScVector::norm_types
  = boost::assign::map_list_of("l1",   NORM_1)
                              ("l2",   NORM_2)
                              ("linf", NORM_INFINITY);

//-----------------------------------------------------------------------------
PETScVector::PETScVector(std::string type)
{
  if (type == "global" && dolfin::MPI::num_processes() > 1)
    init(0, 0, "mpi");
  else
    init(0, 0, "sequential");
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(uint N, std::string type)
{
  if (type == "global")
  {
    // Get local range
    const std::pair<uint, uint> range = MPI::local_range(N);
    if (range.first == 0 && range.second == N)
      init(N, 0, "sequential");
    else
    {
      const uint n = range.second - range.first;
      init(N, n, "mpi");
    }
  }
  else if (type == "local")
    init(0, 0, "sequential");
  else
    error("PETScVector type not known.");
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(boost::shared_ptr<Vec> x): x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v)
{
  *this = v;
}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  // Do nothing. The custom shared_ptr deleter takes care of the cleanup.
}
//-----------------------------------------------------------------------------
void PETScVector::resize(uint N)
{
  if (x && this->size() == N)
    return;

  if (!x)
    error("PETSc vector has not been initialised. Cannot call PETScVector::resize.");

  // Figure out vector type
  std::string type;
  uint n = 0;
  const VecType petsc_type;
  VecGetType(*x, &petsc_type);
  if (strcmp(petsc_type, VECSEQ) == 0)
    type = "sequential";
  else if (strcmp(petsc_type, VECMPI) == 0)
  {
    const std::pair<uint, uint> range = MPI::local_range(N);
    n = range.second - range.first;
    type = "mpi";
  }
  else
    error("Unknown PETSc vector type.");

  // Initialise vector
  init(N, n, type);
}
//-----------------------------------------------------------------------------
PETScVector* PETScVector::copy() const
{
  PETScVector* v = new PETScVector(*this);
  return v;
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(Array<double>& values) const
{
  assert(x);
  if (size() == 0)
    return;

  const uint n0 = local_range().first;
  const uint local_size = local_range().second - local_range().first;
  assert(values.size() >= local_size);

  std::vector<int> rows(local_size);
  for (uint i = 0; i < local_size; ++i)
    rows[i] = i + n0;

  VecGetValues(*x, local_size, &rows[0], values.data().get());
}
//-----------------------------------------------------------------------------
void PETScVector::set_local(const Array<double>& values)
{
  assert(x);
  assert(values.size() == size());
  if (size() == 0)
    return;

  const uint n0 = local_range().first;
  const uint local_size = local_range().second - local_range().first;
  std::vector<int> rows(local_size);
  for (uint i = 0; i < local_size; ++i)
    rows[i] = i + n0;

  VecSetValues(*x, local_size, &rows[0], values.data().get(), INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const Array<double>& values)
{
  assert(x);
  assert(values.size() == size());
  if (size() == 0)
    return;

  const uint n0 = local_range().first;
  const uint local_size = local_range().second - local_range().first;
  std::vector<int> rows(local_size);
  for (uint i = 0; i < local_size; ++i)
    rows[i] = i + n0;

  VecSetValues(*x, local_size, &rows[0], values.data().get(), ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::get(double* block, uint m, const uint* rows) const
{
  assert(x);

  // If vector is local, just get the values. For distributed vectors, perform
  // first a gather into a local vector
  if (local_range().first == 0 && local_range().second == size())
    get_local(block, m, rows);
  else
  {
    PETScVector y("local");
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
    gather(y, indices);

    // Get entries of y
    y.get_local(block, m, &local_indices[0]);
  }
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(double* block, uint m, const uint* rows) const
{
  assert(x);
  int _m = static_cast<int>(m);
  const int* _rows = reinterpret_cast<int*>(const_cast<uint*>(rows));
  if (m == 0)
  {
    _rows = &_m;
    double tmp = 0.0;
    block = &tmp;
  }
  VecGetValues(*x, _m, _rows, block);
}
//-----------------------------------------------------------------------------
void PETScVector::set(const double* block, uint m, const uint* rows)
{
  assert(x);
  int _m =  static_cast<int>(m);
  const int* _rows = reinterpret_cast<int*>(const_cast<uint*>(rows));
  if (m == 0)
  {
    _rows = &_m;
    double tmp = 0;
    block = &tmp;
  }
  VecSetValues(*x, _m, _rows, block, INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::add(const double* block, uint m, const uint* rows)
{
  assert(x);
  int _m =  static_cast<int>(m);
  const int* _rows = reinterpret_cast<int*>(const_cast<uint*>(rows));
  if (m == 0)
  {
    _rows = &_m;
    double tmp = 0;
    block = &tmp;
  }
  VecSetValues(*x, _m, _rows, block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::apply(std::string mode)
{
  assert(x);
  VecAssemblyBegin(*x);
  VecAssemblyEnd(*x);
}
//-----------------------------------------------------------------------------
void PETScVector::zero()
{
  assert(x);
  double a = 0.0;
  VecSet(*x, a);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScVector::size() const
{
  int n = 0;
  if (x)
    VecGetSize(*x, &n);
  return static_cast<uint>(n);
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> PETScVector::local_range() const
{
  std::pair<uint, uint> range;
  VecGetOwnershipRange(*x, (int*) &range.first, (int*) &range.second);
  assert(range.first <= range.second);
  return range;
}
//-----------------------------------------------------------------------------
const GenericVector& PETScVector::operator= (const GenericVector& v)
{
  *this = v.down_cast<PETScVector>();
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (const PETScVector& v)
{
  assert(v.x);

  // Check for self-assignment
  if (this != &v)
  {
    x.reset(new Vec, PETScVectorDeleter());

    // Create new vector
    VecDuplicate(*(v.x), x.get());

    // Copy data
    VecCopy(*(v.x), *x);
  }
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator= (double a)
{
  assert(x);
  VecSet(*x, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator+= (const GenericVector& x)
{
  this->axpy(1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator-= (const GenericVector& x)
{
  this->axpy(-1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator*= (const double a)
{
  assert(x);
  VecScale(*x, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator*= (const GenericVector& y)
{
  assert(x);

  const PETScVector& v = y.down_cast<PETScVector>();
  assert(v.x);

  if (size() != v.size())
    error("The vectors must be of the same size.");

  VecPointwiseMult(*x,*x,*v.x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScVector& PETScVector::operator/= (const double a)
{
  assert(x);
  assert(a != 0.0);

  const double b = 1.0/a;
  VecScale(*x, b);
  return *this;
}
//-----------------------------------------------------------------------------
double PETScVector::inner(const GenericVector& y) const
{
  assert(x);

  const PETScVector& v = y.down_cast<PETScVector>();
  assert(v.x);

  double a;
  VecDot(*(v.x), *x, &a);
  return a;
}
//-----------------------------------------------------------------------------
void PETScVector::axpy(double a, const GenericVector& y)
{
  assert(x);

  const PETScVector& v = y.down_cast<PETScVector>();
  assert(v.x);

  if (size() != v.size())
    error("The vectors must be of the same size.");

  VecAXPY(*x, a, *(v.x));
}
//-----------------------------------------------------------------------------
double PETScVector::norm(std::string norm_type) const
{
  assert(x);
  if (norm_types.count(norm_type) == 0)
    error("Norm type for PETScVector unknown.");

  double value = 0.0;
  VecNorm(*x, norm_types.find(norm_type)->second, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::min() const
{
  assert(x);

  double value = 0.0;
  int position = 0;
  VecMin(*x, &position, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::max() const
{
  assert(x);

  double value = 0.0;
  int position = 0;
  VecMax(*x, &position, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScVector::sum() const
{
  assert(x);

  double value = 0.0;
  VecSum(*x, &value);
  return value;
}
//-----------------------------------------------------------------------------
std::string PETScVector::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScVector not implemented, calling PETSc VecView directly.");

    // Get vector type
    const VecType petsc_type;
    VecGetType(*x, &petsc_type);

    if (strcmp(petsc_type, VECSEQ) == 0)
      VecView(*x, PETSC_VIEWER_STDOUT_SELF);
    else
      VecView(*x, PETSC_VIEWER_STDOUT_WORLD);
  }
  else
  {
    s << "<PETScVector of size " << size() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void PETScVector::gather(GenericVector& y,
                         const std::vector<uint>& indices) const
{
  assert(x);

  // Down cast to a PETScVector
  PETScVector& _y = y.down_cast<PETScVector>();

  // Check that y is a local vector
  const VecType petsc_type;
  VecGetType(*(_y.vec()), &petsc_type);
  if (strcmp(petsc_type, VECSEQ) != 0)
    error("PETScVector::gather can only gather into local vectors");

  // Prepare data for index sets
  const int* global_indices = reinterpret_cast<int*>(const_cast<uint*>(&indices[0]));
  const int n = indices.size();
  std::vector<int> local_indices;
  local_indices.reserve(n);
  for (int i = 0; i < n; ++i)
    local_indices.push_back(i);

  // PETSc will bail out if it receives a NULL pointer even though m == 0.
  // Can't return from function as this will cause a lock up in parallel
  if (n == 0)
    global_indices = &n;

  // Create index sets
  IS from, to;
  ISCreateGeneral(PETSC_COMM_SELF, n, global_indices,    &from);
  ISCreateGeneral(PETSC_COMM_SELF, n, &local_indices[0], &to);

  // Resize vector if required
  y.resize(n);

  // Perform scatter
  VecScatter scatter;
  VecScatterCreate(*x, from, *(_y.vec()), to, &scatter);
  VecScatterBegin(scatter, *x, *(_y.vec()), INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter,   *x, *(_y.vec()), INSERT_VALUES, SCATTER_FORWARD);

  // Clean up
  ISDestroy(from);
  ISDestroy(to);
  VecScatterDestroy(scatter);
}
//-----------------------------------------------------------------------------
void PETScVector::init(uint N, uint n, std::string type)
{
  // Create vector
  if (x && !x.unique())
    error("Cannot init/resize PETScVector. More than one object points to the underlying PETSc object.");
  x.reset(new Vec, PETScVectorDeleter());

  // Initialize vector, either default or MPI vector
  if (type == "sequential")
  {
    VecCreateSeq(PETSC_COMM_SELF, N, x.get());
    VecSetFromOptions(*x);
  }
  else if (type == "mpi")
    VecCreateMPI(PETSC_COMM_WORLD, n, N, x.get());
  else
    error("Unknown vector type in PETScVector::init.");
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Vec> PETScVector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& PETScVector::factory() const
{
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------

#endif
