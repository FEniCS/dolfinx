// Copyright (C) 2004-2007 Johan Hoffman, Johan Jansson and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2005-2010.
// Modified by Martin Sandve Alnes 2008
// Modified by Johannes Ring, 2011.
// Modified by Fredrik Valdmanis, 2011
//
// First added:  2004
// Last changed: 2011-09-29

#ifdef HAS_PETSC

#include <cmath>
#include <numeric>
#include <boost/assign/list_of.hpp>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScVector.h"
#include "uBLASVector.h"
#include "PETScFactory.h"
#include "PETScCuspFactory.h"
#include <dolfin/common/MPI.h>

using namespace dolfin;

const std::map<std::string, NormType> PETScVector::norm_types
  = boost::assign::map_list_of("l1",   NORM_1)
                              ("l2",   NORM_2)
                              ("linf", NORM_INFINITY);

//-----------------------------------------------------------------------------
PETScVector::PETScVector(std::string type, std::string vector_arch) : arch(vector_arch)
{
  if (type != "global" && type != "local")
    error("PETSc vector type unknown.");

#ifndef HAS_PETSC_CUSP
  if (vector_arch == "gpu")
    error("PETSc not compiled with Cusp support, cannot create GPU vector");
#endif

  if (vector_arch != "cpu" && vector_arch != "gpu")
    error("PETSc vector architechture unknown.");

  // Empty ghost indices vector
  const std::vector<uint> ghost_indices;

  // Trivial range
  const std::pair<uint, uint> range(0, 0);

  if (type == "global" && dolfin::MPI::num_processes() > 1)
    init(range, ghost_indices, true);
  else
    init(range, ghost_indices, false);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(uint N, std::string type, std::string vector_arch) : arch(vector_arch)
{
#ifndef HAS_PETSC_CUSP
  if (vector_arch == "gpu")
    error("PETSc not compiled with Cusp support, cannot create GPU vector");
#endif
  
  // Empty ghost indices vector
  const std::vector<uint> ghost_indices;

  if (type == "global")
  {
    // Compute a local range
    const std::pair<uint, uint> range = MPI::local_range(N);

    if (range.first == 0 && range.second == N)
      init(range, ghost_indices, false);
    else
      init(range, ghost_indices, true);
  }
  else if (type == "local")
  {
    const std::pair<uint, uint> range(0, N);
    init(range, ghost_indices, false);
  }
  else
    error("PETScVector type not known.");
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const GenericSparsityPattern& sparsity_pattern): arch("cpu")
{
  std::vector<uint> ghost_indices;
  resize(sparsity_pattern.local_range(0), ghost_indices);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(boost::shared_ptr<Vec> x): x(x), arch("cpu")
{
  // Do nothing else
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v): arch("cpu")
{
  *this = v;
}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  // Do nothing. The custom shared_ptr deleter takes care of the cleanup.
}
//-----------------------------------------------------------------------------
bool PETScVector::distributed() const
{
  assert(x);

  // Get type
  const VecType petsc_type;
  VecGetType(*x, &petsc_type);

  // Return type
  bool _distributed = false;
  if (strcmp(petsc_type, VECMPI) == 0)
    _distributed = true;
  else if (strcmp(petsc_type, VECSEQ) == 0)
    _distributed =  false;
#ifdef HAS_PETSC_CUSP
  else if (strcmp(petsc_type, VECSEQCUSP) == 0)
    _distributed = false;
#endif
  else
    error("Unknown PETSc vector type.");

  return _distributed;
}
//-----------------------------------------------------------------------------
void PETScVector::resize(uint N)
{
  if (x && this->size() == N)
    return;

  if (!x)
    error("PETSc vector has not been initialised. Cannot call PETScVector::resize.");

  // Get vector type
  const bool _distributed = distributed();

  // Create vector
  if (_distributed)
  {
    const std::pair<uint, uint> range = MPI::local_range(N);
    resize(range);
  }
  else
  {
    const std::pair<uint, uint> range(0, N);
    resize(range);
  }
}
//-----------------------------------------------------------------------------
void PETScVector::resize(std::pair<uint, uint> range)
{
  // Create empty ghost indices vector
  std::vector<uint> ghost_indices;
  resize(range, ghost_indices);
}
//-----------------------------------------------------------------------------
void PETScVector::resize(std::pair<uint, uint> range,
                         const std::vector<uint>& ghost_indices)
{
  // Get local size
  assert(range.second - range.first >= 0);

  // FIXME: Can this check be made robust? Need to avoid parallel lock-up.
  //        Cannot just check size because range may change.
  // Check if resizing is required
  //if (x && (this->local_range().first == range.first && this->local_range().second == range.second))
  //  return;

  // Get type
  const bool _distributed = distributed();

  // Re-initialise vector
  init(range, ghost_indices, _distributed);
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
  const uint n0 = local_range().first;
  const uint local_size = local_range().second - local_range().first;
  values.resize(local_size);

  if (local_size == 0)
    return;

  std::vector<int> rows(local_size);
  for (uint i = 0; i < local_size; ++i)
    rows[i] = i + n0;

  VecGetValues(*x, local_size, &rows[0], values.data().get());
}
//-----------------------------------------------------------------------------
void PETScVector::set_local(const Array<double>& values)
{
  assert(x);
  const uint n0 = local_range().first;
  const uint local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
    error("PETScVector::set_local: length of values array is not equal to local vector size.");

  if (local_size == 0)
    return;

  // Build array of global indices
  std::vector<int> rows(local_size);
  for (uint i = 0; i < local_size; ++i)
    rows[i] = i + n0;

  VecSetValues(*x, local_size, &rows[0], values.data().get(), INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const Array<double>& values)
{
  assert(x);
  const uint n0 = local_range().first;
  const uint local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
    error("PETScVector::add_local: length of values array is not equal to local vector size.");

  if (local_size == 0)
    return;

  // Build array of global indices
  std::vector<int> rows(local_size);
  for (uint i = 0; i < local_size; ++i)
    rows[i] = i + n0;

  VecSetValues(*x, local_size, &rows[0], values.data().get(), ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(double* block, uint m, const uint* rows) const
{
  assert(x);
  int _m = static_cast<int>(m);
  const int* _rows = reinterpret_cast<const int*>(rows);

  // Handle case that m = 0 (VecGetValues is collective -> must be called be
  //                         all processes)
  if (m == 0)
  {
    _rows = &_m;
    double tmp = 0.0;
    block = &tmp;
  }

  // Use VecGetValues if no ghost points, otherwise check for ghost values
  if (ghost_global_to_local.size() == 0 || m == 0)
  {
    VecGetValues(*x, _m, _rows, block);
  }
  else
  {
    assert(x_ghosted);

    // Get local range
    const uint n0 = local_range().first;
    const uint n1 = local_range().second;
    const uint local_size = n1 - n0;

    // Build list of rows, and get from ghosted vector
    std::vector<int> local_rows(m);
    for (uint i = 0; i < m; ++i)
    {
      if (rows[i] >= n0 && rows[i] < n1)
        local_rows[i] = rows[i] - n0;
      else
      {
        boost::unordered_map<uint, uint>::const_iterator local_index = ghost_global_to_local.find(rows[i]);
        assert(local_index != ghost_global_to_local.end());
        local_rows[i] = local_index->second + local_size;
      }
    }

    // Pick values from ghosted vector
    VecGetValues(*x_ghosted, _m, &local_rows[0], block);
  }
}
//-----------------------------------------------------------------------------
void PETScVector::set(const double* block, uint m, const uint* rows)
{
  assert(x);

  if (m == 0)
    return;

  VecSetValues(*x, m, reinterpret_cast<const int*>(rows), block, INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScVector::add(const double* block, uint m, const uint* rows)
{
  assert(x);

  if (m == 0)
    return;

  VecSetValues(*x, m, reinterpret_cast<const int*>(rows), block, ADD_VALUES);
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
dolfin::uint PETScVector::local_size() const
{
  int n = 0;
  if (x)
    VecGetLocalSize(*x, &n);
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
bool PETScVector::owns_index(uint i) const
{
  if (i >= local_range().first && i < local_range().second)
    return true;
  else
    return false;
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
    x.reset(new Vec(0), PETScVectorDeleter());

    // Create new vector
    VecDuplicate(*(v.x), x.get());

    // Copy data
    VecCopy(*(v.x), *x);

    // Copy ghost data
    this->ghost_global_to_local = v.ghost_global_to_local;

    // Create ghost view
    this->x_ghosted.reset(new Vec(0), PETScVectorDeleter());
    if (ghost_global_to_local.size() > 0)
      VecGhostGetLocalForm(*x, x_ghosted.get());
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
void PETScVector::update_ghost_values()
{
  VecGhostUpdateBegin(*x, INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(*x, INSERT_VALUES, SCATTER_FORWARD);
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
    error("The vectors must be of the same for point-wise multiplication size.");

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

  const PETScVector& _y = y.down_cast<PETScVector>();
  assert(_y.x);

  double a;
  VecDot(*(_y.x), *x, &a);
  return a;
}
//-----------------------------------------------------------------------------
void PETScVector::axpy(double a, const GenericVector& y)
{
  assert(x);

  const PETScVector& _y = y.down_cast<PETScVector>();
  assert(_y.x);

  if (size() != _y.size())
    error("The vectors must be of the same size for addition.");

  VecAXPY(*x, a, *(_y.x));
}
//-----------------------------------------------------------------------------
void PETScVector::abs()
{
  assert(x);
  VecAbs(*x);
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
double PETScVector::sum(const Array<uint>& rows) const
{
  assert(x);
  const uint n0 = local_range().first;
  const uint n1 = local_range().second;

  // Build sets of local and nonlocal entries
  Set<uint> local_rows;
  Set<uint> send_nonlocal_rows;
  for (uint i = 0; i < rows.size(); ++i)
  {
    if (rows[i] >= n0 && rows[i] < n1)
      local_rows.insert(rows[i]);
    else
      send_nonlocal_rows.insert(rows[i]);
  }

  // Send nonlocal rows indices to other processes
  const uint num_processes  = MPI::num_processes();
  const uint process_number = MPI::process_number();
  for (uint i = 1; i < num_processes; ++i)
  {
    // Receive data from process p - i (i steps to the left), send data to
    // process p + i (i steps to the right)
    const uint source = (process_number - i + num_processes) % num_processes;
    const uint dest   = (process_number + i) % num_processes;

    // Send and receive data
    std::vector<uint> received_nonlocal_rows;
    MPI::send_recv(send_nonlocal_rows.set(), dest,
                   received_nonlocal_rows, source);

    // Add rows which reside on this process
    for (uint j = 0; j < received_nonlocal_rows.size(); ++j)
    {
      if (received_nonlocal_rows[j] >= n0 && received_nonlocal_rows[j] < n1)
        local_rows.insert(received_nonlocal_rows[j]);
    }
  }

  // Get local values
  std::vector<double> local_values(local_rows.size());
  get_local(&local_values[0], local_rows.size(), &local_rows.set()[0]);

  // Compute local sum
  const double local_sum = std::accumulate(local_values.begin(), local_values.end(), 0.0);

  return MPI::sum(local_sum);
}
//-----------------------------------------------------------------------------
std::string PETScVector::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    //warning("Verbose output for PETScVector not implemented, calling PETSc VecView directly.");

    // Get vector type
    const VecType petsc_type;
    VecGetType(*x, &petsc_type);

    if (strcmp(petsc_type, VECSEQ) == 0)
      VecView(*x, PETSC_VIEWER_STDOUT_SELF);
    else
      VecView(*x, PETSC_VIEWER_STDOUT_WORLD);
  }
  else
    s << "<PETScVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void PETScVector::gather(GenericVector& y, const Array<uint>& indices) const
{
  assert(x);

  // Down cast to a PETScVector
  PETScVector& _y = y.down_cast<PETScVector>();

  // Check that y is a local vector
  const VecType petsc_type;
  VecGetType(*(_y.vec()), &petsc_type);
  if (strcmp(petsc_type, VECSEQ) != 0)
    error("PETScVector::gather can only gather into local vectors");
#ifdef HAS_PETSC_CUSP
  else if (strcmp(petsc_type, VECSEQCUSP) != 0)
    error("PETScVector::gather can only gather into local vectors");
#endif


  // Prepare data for index sets (global indices)
  const int* global_indices = reinterpret_cast<const int*>(indices.data().get());

  // Prepare data for index sets (local indices)
  const int n = indices.size();

  // PETSc will bail out if it receives a NULL pointer even though m == 0.
  // Can't return from function since function calls are collective.
  if (n == 0)
    global_indices = &n;

  // Create local index sets
  IS from, to;
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 1
  ISCreateGeneral(PETSC_COMM_SELF, n, global_indices, PETSC_COPY_VALUES, &from);
  #else
  ISCreateGeneral(PETSC_COMM_SELF, n, global_indices,    &from);
  #endif
  ISCreateStride(PETSC_COMM_SELF, n, 0 , 1, &to);

  // Resize vector if required
  y.resize(n);

  // Perform scatter
  VecScatter scatter;
  VecScatterCreate(*x, from, *(_y.vec()), to, &scatter);
  VecScatterBegin(scatter, *x, *(_y.vec()), INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter,   *x, *(_y.vec()), INSERT_VALUES, SCATTER_FORWARD);

  // Clean up
#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 1
  ISDestroy(from);
  ISDestroy(to);
  VecScatterDestroy(scatter);
#else
  ISDestroy(&from);
  ISDestroy(&to);
  VecScatterDestroy(&scatter);
#endif
}
//-----------------------------------------------------------------------------
void PETScVector::gather(Array<double>& x, const Array<uint>& indices) const
{
  x.resize(indices.size());
  PETScVector y("local");
  gather(y, indices);
  assert(y.local_size() == x.size());

  y.get_local(x);

  double sum = 0.0;
  for (uint i = 0; i < x.size(); ++i)
    sum += x[i]*x[i];
}
//-----------------------------------------------------------------------------
void PETScVector::gather_on_zero(Array<double>& x) const
{
  if (MPI::process_number() == 0)
    x.resize(size());
  else
    x.resize(0);

  boost::shared_ptr<Vec> vout(new Vec);
  VecScatter scatter;
  VecScatterCreateToZero(*this->x, &scatter, vout.get());

  VecScatterBegin(scatter, *this->x, *vout, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter, *this->x, *vout, INSERT_VALUES, SCATTER_FORWARD);

#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 1
  VecScatterDestroy(scatter);
#else
  VecScatterDestroy(&scatter);
#endif

  // Wrap PETSc vector
  if (MPI::process_number() == 0)
  {
    PETScVector _vout(vout);
    _vout.get_local(x);
  }
}
//-----------------------------------------------------------------------------
void PETScVector::init(std::pair<uint, uint> range,
                       const std::vector<uint>& ghost_indices, bool distributed)
{
  // Create vector
  if (x && !x.unique())
    error("Cannot init/resize PETScVector. More than one object points to the underlying PETSc object.");
  x.reset(new Vec(0), PETScVectorDeleter());

  const uint local_size = range.second - range.first;
  assert(range.second - range.first >= 0);

  // Initialize vector, either default or MPI vector
  if (!distributed)
  {
    // FIXME: Make it look better!
    VecCreate(PETSC_COMM_SELF, x.get());
    // Set type to be either standard or Cusp sequential vector
    if (arch == "cpu")
      VecSetType(*x, VECSEQ);
#ifdef HAS_PETSC_CUSP
    else if (arch == "gpu")
      VecSetType(*x, VECSEQCUSP);
#endif
    else 
      error("PETSc vector architecture unknown");
    
    VecSetSizes(*x, local_size, PETSC_DECIDE);
    VecSetFromOptions(*x);

  }
  else
  {
    // TODO: Implement VECMPICUSP vectors
    if (arch == "gpu")
      error("Distributed PETSc Cusp vectors not implemented yet.");
    
    // Clear ghost indices map
    ghost_global_to_local.clear();

    const int* _ghost_indices = 0;
    if (ghost_indices.size() > 0)
      _ghost_indices = reinterpret_cast<const int*>(&ghost_indices[0]);

    VecCreateGhost(PETSC_COMM_WORLD, local_size, PETSC_DECIDE,
                   ghost_indices.size(), _ghost_indices, x.get());

    // Build global-to-local map for ghost indices
    for (uint i = 0; i < ghost_indices.size(); ++i)
      ghost_global_to_local.insert(std::pair<uint, uint>(ghost_indices[i], i));

    // Create ghost view
    x_ghosted.reset(new Vec(0), PETScVectorDeleter());
    VecGhostGetLocalForm(*x, x_ghosted.get());
  }
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Vec> PETScVector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
void PETScVector::reset()
{
  x.reset();
  x_ghosted.reset();
  ghost_global_to_local.clear();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& PETScVector::factory() const
{
  if (arch == "cpu")
    return PETScFactory::instance();
#ifdef HAS_PETSC_CUSP
  else if (arch == "gpu")
    return PETScCuspFactory::instance();
#endif
  else
    error("PETSc vector architecture unknown/unsupported");

  // Return something to keep the compiler happy. Code will never be reached.
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------
#endif
