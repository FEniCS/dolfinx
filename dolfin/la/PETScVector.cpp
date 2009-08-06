// Copyright (C) 2004-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2009.
// Modified by Martin Sandve Alnes 2008
//
// First added:  2004
// Last changed: 2008-05-22

#ifdef HAS_PETSC

#include <cmath>
#include <boost/assign/list_of.hpp>
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
PETScVector::PETScVector():
    Variable("x", "a sparse vector"),
    x(static_cast<Vec*>(0), PETScVectorDeleter())
{
  if (dolfin::MPI::num_processes() > 1)
    init(0, 0, "mpi");
  else
    init(0, 0, "sequential");
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(uint N):
    Variable("x", "a sparse vector"),
    x(static_cast<Vec*>(0), PETScVectorDeleter())
{
  // FIXME: type should be passed as an argument
  std::string type = "global";
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
PETScVector::PETScVector(boost::shared_ptr<Vec> x):
    Variable("x", "a vector"),
    x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v):
    Variable("x", "a vector"),
    x(static_cast<Vec*>(0), PETScVectorDeleter())
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
  #if PETSC_VERSION_MAJOR > 2
  const VecType petsc_type;
  #else
  VecType petsc_type;
  #endif
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
void PETScVector::get(double* values) const
{
  assert(x);

  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecGetValues(*x, m, rows, values);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::set(double* values)
{
  assert(x);

  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecSetValues(*x, m, rows, values, INSERT_VALUES);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::add(double* values)
{
  assert(x);

  int m = static_cast<int>(size());
  int* rows = new int[m];
  for (int i = 0; i < m; i++)
    rows[i] = i;

  VecSetValues(*x, m, rows, values, ADD_VALUES);

  delete [] rows;
}
//-----------------------------------------------------------------------------
void PETScVector::get(double* block, uint m, const uint* rows) const
{
  assert(x);
  VecGetValues(*x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block);
}
//-----------------------------------------------------------------------------
void PETScVector::set(const double* block, uint m, const uint* rows)
{
  assert(x);
  VecSetValues(*x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block,
               INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::add(const double* block, uint m, const uint* rows)
{
  assert(x);
  VecSetValues(*x, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), block,
               ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::apply()
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
std::pair<dolfin::uint, dolfin::uint> PETScVector::ownership_range() const
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
    boost::shared_ptr<Vec> _x(new Vec, PETScVectorDeleter());
    x = _x;

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
void PETScVector::disp(uint precision) const
{
  // Get vector type
  #if PETSC_VERSION_MAJOR > 2
  const VecType petsc_type;
  #else
  VecType petsc_type;
  #endif
  VecGetType(*x, &petsc_type);

  if (strcmp(petsc_type, VECSEQ) == 0)
    VecView(*x, PETSC_VIEWER_STDOUT_SELF);	 
  else
    VecView(*x, PETSC_VIEWER_STDOUT_WORLD);	 
}
//-----------------------------------------------------------------------------
boost::shared_ptr<PETScVector> PETScVector::gather(const std::vector<uint>& indices) const
{
  const int n = indices.size();
  int* local_indices  = new int[n];
  int* global_indices = new int[n];

  for (int i = 0; i < n; ++i)
  {
    global_indices[i] = indices[i];      
    local_indices[i]  = i;      
  }

  // Create index sets
  IS from, to;
  ISCreateGeneral(PETSC_COMM_SELF, n, global_indices, &from);
  ISCreateGeneral(PETSC_COMM_SELF, n, local_indices,  &to);

  // Create local PETSc vector
  boost::shared_ptr<Vec> a_vec(new Vec, PETScVectorDeleter());
  VecCreateSeq(PETSC_COMM_SELF, n, a_vec.get());

  // Perform scatter
  VecScatter scatter;
  VecScatterCreate(*x, from, *a_vec, to, &scatter);
  VecScatterBegin(scatter, *x, *a_vec, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter,   *x, *a_vec, INSERT_VALUES, SCATTER_FORWARD);

  // Clean up
  ISDestroy(from);
  ISDestroy(to);
  VecScatterDestroy(scatter);
  delete [] local_indices;
  delete [] global_indices;

  // Create PETScVector
  boost::shared_ptr<PETScVector> a(new PETScVector(a_vec)); 
  return a;
}
//-----------------------------------------------------------------------------
void PETScVector::init(uint N, uint n, std::string type)
{
  // Create vector
  if (!x.unique())
    error("Cannot init/resize PETScVector. More than one object points to the underlying PETSc object.");
  boost::shared_ptr<Vec> _x(new Vec, PETScVectorDeleter());
  x = _x;

  // Initialize vector, either default or MPI vector
  if (type == "sequential")
  {
    VecCreateSeq(PETSC_COMM_SELF, N, x.get());
    VecSetFromOptions(*x);
  }
  else if (type == "mpi")
  {
    //assert(n > 0);
    VecCreateMPI(PETSC_COMM_WORLD, n, N, x.get());
  }
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
LogStream& dolfin::operator<< (LogStream& stream, const PETScVector& x)
{
  stream << "[ PETSc vector of size " << x.size() << " ]";
  return stream;
}
//-----------------------------------------------------------------------------

#endif
