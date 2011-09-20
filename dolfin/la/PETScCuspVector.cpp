// Copyright (C) 2011 Fredrik Valdmanis 
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
// First added:  2011-09-13
// Last changed: 2011-09-13

//#ifdef PETSC_HAVE_CUSP // FIXME: Find a functioning test

#include <cmath>
#include <numeric>
#include <boost/assign/list_of.hpp>
#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScCuspVector.h"
#include "uBLASVector.h"
#include "PETScCuspFactory.h"
#include <dolfin/common/MPI.h>

using namespace dolfin;

const std::map<std::string, NormType> PETScCuspVector::norm_types
  = boost::assign::map_list_of("l1",   NORM_1)
                              ("l2",   NORM_2)
                              ("linf", NORM_INFINITY);

//-----------------------------------------------------------------------------
PETScCuspVector::PETScCuspVector(std::string type)
{
  if (type != "global" && type != "local")
    error("PETSc vector type unknown.");

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
PETScCuspVector::PETScCuspVector(uint N, std::string type)
{
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
    error("PETScCuspVector type not known.");
}
//-----------------------------------------------------------------------------
PETScCuspVector::PETScCuspVector(const GenericSparsityPattern& sparsity_pattern)
{
  std::vector<uint> ghost_indices;
  resize(sparsity_pattern.local_range(0), ghost_indices);
}
//-----------------------------------------------------------------------------
PETScCuspVector::PETScCuspVector(boost::shared_ptr<Vec> x): x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScCuspVector::PETScCuspVector(const PETScCuspVector& v)
{
  *this = v;
}
//-----------------------------------------------------------------------------
PETScCuspVector::~PETScCuspVector()
{
  // Do nothing. The custom shared_ptr deleter takes care of the cleanup.
}
//-----------------------------------------------------------------------------
bool PETScCuspVector::distributed() const
{
  assert(x);

  // Get type
  const VecType petsc_type;
  VecGetType(*x, &petsc_type);

  // Return type
  bool _distributed = false;
  if (strcmp(petsc_type, VECMPI) == 0)
    _distributed = true;
  else if (strcmp(petsc_type, VECSEQCUSP) == 0)
    _distributed =  false;
  else
    // FIXME: Output the type of the vector
    error("Unknown PETSc vector type.");

  return _distributed;
}
//-----------------------------------------------------------------------------
void PETScCuspVector::resize(uint N)
{
  if (x && this->size() == N)
    return;

  if (!x)
    error("PETSc Cusp vector has not been initialised. Cannot call PETScCuspVector::resize.");

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
void PETScCuspVector::resize(std::pair<uint, uint> range)
{
  // Create empty ghost indices vector
  std::vector<uint> ghost_indices;
  resize(range, ghost_indices);
}
//-----------------------------------------------------------------------------
void PETScCuspVector::resize(std::pair<uint, uint> range,
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
PETScCuspVector* PETScCuspVector::copy() const
{
  PETScCuspVector* v = new PETScCuspVector(*this);
  return v;
}
//-----------------------------------------------------------------------------
void PETScCuspVector::get_local(Array<double>& values) const
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
void PETScCuspVector::set_local(const Array<double>& values)
{
  assert(x);
  const uint n0 = local_range().first;
  const uint local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
    error("PETScCuspVector::set_local: length of values array is not equal to local vector size.");

  if (local_size == 0)
    return;

  // Build array of global indices
  std::vector<int> rows(local_size);
  for (uint i = 0; i < local_size; ++i)
    rows[i] = i + n0;

  VecSetValues(*x, local_size, &rows[0], values.data().get(), INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScCuspVector::add_local(const Array<double>& values)
{
  assert(x);
  const uint n0 = local_range().first;
  const uint local_size = local_range().second - local_range().first;
  if (values.size() != local_size)
    error("PETScCuspVector::add_local: length of values array is not equal to local vector size.");

  if (local_size == 0)
    return;

  // Build array of global indices
  std::vector<int> rows(local_size);
  for (uint i = 0; i < local_size; ++i)
    rows[i] = i + n0;

  VecSetValues(*x, local_size, &rows[0], values.data().get(), ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScCuspVector::get_local(double* block, uint m, const uint* rows) const
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
void PETScCuspVector::set(const double* block, uint m, const uint* rows)
{
  assert(x);

  if (m == 0)
    return;

  VecSetValues(*x, m, reinterpret_cast<const int*>(rows), block, INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScCuspVector::add(const double* block, uint m, const uint* rows)
{
  assert(x);

  if (m == 0)
    return;

  VecSetValues(*x, m, reinterpret_cast<const int*>(rows), block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScCuspVector::apply(std::string mode)
{
  assert(x);
  VecAssemblyBegin(*x);
  VecAssemblyEnd(*x);
}
//-----------------------------------------------------------------------------
void PETScCuspVector::zero()
{
  assert(x);
  double a = 0.0;
  VecSet(*x, a);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScCuspVector::size() const
{
  int n = 0;
  if (x)
    VecGetSize(*x, &n);
  return static_cast<uint>(n);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScCuspVector::local_size() const
{
  int n = 0;
  if (x)
    VecGetLocalSize(*x, &n);
  return static_cast<uint>(n);
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> PETScCuspVector::local_range() const
{
  std::pair<uint, uint> range;
  VecGetOwnershipRange(*x, (int*) &range.first, (int*) &range.second);
  assert(range.first <= range.second);
  return range;
}
//-----------------------------------------------------------------------------
bool PETScCuspVector::owns_index(uint i) const
{
  if (i >= local_range().first && i < local_range().second)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
const GenericVector& PETScCuspVector::operator= (const GenericVector& v)
{
  *this = v.down_cast<PETScCuspVector>();
  return *this;
}
//-----------------------------------------------------------------------------
const PETScCuspVector& PETScCuspVector::operator= (const PETScCuspVector& v)
{
  assert(v.x);

  // Check for self-assignment
  if (this != &v)
  {
    x.reset(new Vec(0), PETScCuspVectorDeleter());

    // Create new vector
    VecDuplicate(*(v.x), x.get());

    // Copy data
    VecCopy(*(v.x), *x);

    // Copy ghost data
    this->ghost_global_to_local = v.ghost_global_to_local;

    // Create ghost view
    this->x_ghosted.reset(new Vec(0), PETScCuspVectorDeleter());
    if (ghost_global_to_local.size() > 0)
      VecGhostGetLocalForm(*x, x_ghosted.get());
  }
  return *this;
}
//-----------------------------------------------------------------------------
const PETScCuspVector& PETScCuspVector::operator= (double a)
{
  assert(x);
  VecSet(*x, a);
  return *this;
}
//-----------------------------------------------------------------------------
void PETScCuspVector::update_ghost_values()
{
  VecGhostUpdateBegin(*x, INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(*x, INSERT_VALUES, SCATTER_FORWARD);
}
//-----------------------------------------------------------------------------
const PETScCuspVector& PETScCuspVector::operator+= (const GenericVector& x)
{
  this->axpy(1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScCuspVector& PETScCuspVector::operator-= (const GenericVector& x)
{
  this->axpy(-1.0, x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScCuspVector& PETScCuspVector::operator*= (const double a)
{
  assert(x);
  VecScale(*x, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScCuspVector& PETScCuspVector::operator*= (const GenericVector& y)
{
  assert(x);

  const PETScCuspVector& v = y.down_cast<PETScCuspVector>();
  assert(v.x);

  if (size() != v.size())
    error("The vectors must be of the same for point-wise multiplication size.");

  VecPointwiseMult(*x,*x,*v.x);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScCuspVector& PETScCuspVector::operator/= (const double a)
{
  assert(x);
  assert(a != 0.0);

  const double b = 1.0/a;
  VecScale(*x, b);
  return *this;
}
//-----------------------------------------------------------------------------
double PETScCuspVector::inner(const GenericVector& y) const
{
  assert(x);

  const PETScCuspVector& _y = y.down_cast<PETScCuspVector>();
  assert(_y.x);

  double a;
  VecDot(*(_y.x), *x, &a);
  return a;
}
//-----------------------------------------------------------------------------
void PETScCuspVector::axpy(double a, const GenericVector& y)
{
  assert(x);

  const PETScCuspVector& _y = y.down_cast<PETScCuspVector>();
  assert(_y.x);

  if (size() != _y.size())
    error("The vectors must be of the same size for addition.");

  VecAXPY(*x, a, *(_y.x));
}
//-----------------------------------------------------------------------------
void PETScCuspVector::abs()
{
  assert(x);
  VecAbs(*x);
}
//-----------------------------------------------------------------------------
double PETScCuspVector::norm(std::string norm_type) const
{
  assert(x);
  if (norm_types.count(norm_type) == 0)
    error("Norm type for PETScCuspVector unknown.");

  double value = 0.0;
  VecNorm(*x, norm_types.find(norm_type)->second, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScCuspVector::min() const
{
  assert(x);

  double value = 0.0;
  int position = 0;
  VecMin(*x, &position, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScCuspVector::max() const
{
  assert(x);

  double value = 0.0;
  int position = 0;
  VecMax(*x, &position, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScCuspVector::sum() const
{
  assert(x);

  double value = 0.0;
  VecSum(*x, &value);
  return value;
}
//-----------------------------------------------------------------------------
double PETScCuspVector::sum(const Array<uint>& rows) const
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

    // Size of send and receive data
    uint send_buffer_size = send_nonlocal_rows.size();
    uint recv_buffer_size = 0;
    MPI::send_recv(&send_buffer_size, 1, dest, &recv_buffer_size, 1, source);

    // Send and receive data
    std::vector<uint> received_nonlocal_rows(recv_buffer_size);
    MPI::send_recv(&(send_nonlocal_rows.set())[0], send_buffer_size, dest,
                   &received_nonlocal_rows[0], recv_buffer_size, source);

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
std::string PETScCuspVector::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    //warning("Verbose output for PETScCuspVector not implemented, calling PETSc VecView directly.");

    // Get vector type
    const VecType petsc_type;
    VecGetType(*x, &petsc_type);

    if (strcmp(petsc_type, VECSEQ) == 0)
      VecView(*x, PETSC_VIEWER_STDOUT_SELF);
    else
      VecView(*x, PETSC_VIEWER_STDOUT_WORLD);
  }
  else
    s << "<PETScCuspVector of size " << size() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void PETScCuspVector::gather(GenericVector& y, const Array<uint>& indices) const
{
  assert(x);

  // Down cast to a PETScCuspVector
  PETScCuspVector& _y = y.down_cast<PETScCuspVector>();

  // Check that y is a local vector
  const VecType petsc_type;
  VecGetType(*(_y.vec()), &petsc_type);
  if (strcmp(petsc_type, VECSEQ) != 0)
    error("PETScCuspVector::gather can only gather into local vectors");

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
void PETScCuspVector::gather(Array<double>& x, const Array<uint>& indices) const
{
  x.resize(indices.size());
  PETScCuspVector y("local");
  gather(y, indices);
  assert(y.local_size() == x.size());

  y.get_local(x);

  double sum = 0.0;
  for (uint i = 0; i < x.size(); ++i)
    sum += x[i]*x[i];
}
//-----------------------------------------------------------------------------
void PETScCuspVector::gather_on_zero(Array<double>& x) const
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
    PETScCuspVector _vout(vout);
    _vout.get_local(x);
  }
}
//-----------------------------------------------------------------------------
void PETScCuspVector::init(std::pair<uint, uint> range,
                       const std::vector<uint>& ghost_indices, bool distributed)
{
  // Create vector
  if (x && !x.unique())
    error("Cannot init/resize PETScCuspVector. More than one object points to the underlying PETSc object.");
  x.reset(new Vec(0), PETScCuspVectorDeleter());

  const uint local_size = range.second - range.first;
  assert(range.second - range.first >= 0);

  // Initialize vector
  if (!distributed)
  {
    // Initialize vector as sequential Cusp vector
    VecCreate(PETSC_COMM_SELF, x.get());
    VecSetType(*x, VECSEQCUSP);
    VecSetSizes(*x, local_size, PETSC_DECIDE);
    VecSetFromOptions(*x);
  }
  else
  {
    // FIXME: Can we have ghosted vectors of VECMPICUSP type?
    error("Cannot create distributed PETScCusp vector as of yet.");
    /*
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
    x_ghosted.reset(new Vec(0), PETScCuspVectorDeleter());
    VecGhostGetLocalForm(*x, x_ghosted.get());
    */
  }
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Vec> PETScCuspVector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
void PETScCuspVector::reset()
{
  x.reset();
  x_ghosted.reset();
  ghost_global_to_local.clear();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& PETScCuspVector::factory() const
{
  return PETScCuspFactory::instance();
}
//-----------------------------------------------------------------------------
//#endif
