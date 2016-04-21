// Copyright (C) 2007 Magnus Vikstrøm
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
// Modified by Garth N. Wells 2007-20!2
// Modified by Anders Logg 2007-2011
// Modified by Ola Skavhaug 2008-2009
// Modified by Niclas Jansson 2009
// Modified by Joachim B Haga 2012
// Modified by Martin Sandve Alnes 2014

#include <numeric>
#include <algorithm>
#include "SubSystemsManager.h"
#include "MPI.h"

#ifdef HAS_MPI

//-----------------------------------------------------------------------------
dolfin::MPIInfo::MPIInfo()
{
  MPI_Info_create(&info);
}
//-----------------------------------------------------------------------------
dolfin::MPIInfo::~MPIInfo()
{
  MPI_Info_free(&info);
}
//-----------------------------------------------------------------------------
MPI_Info& dolfin::MPIInfo::operator*()
{
  return info;
}
//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
unsigned int dolfin::MPI::rank(const MPI_Comm comm)
{
#ifdef HAS_MPI
  SubSystemsManager::init_mpi();
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
#else
  return 0;
#endif
}
//-----------------------------------------------------------------------------
unsigned int dolfin::MPI::size(const MPI_Comm comm)
{
#ifdef HAS_MPI
  SubSystemsManager::init_mpi();
  int size;
  MPI_Comm_size(comm, &size);
  return size;
#else
  return 1;
#endif
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::is_broadcaster(const MPI_Comm comm)
{
  // Always broadcast from processor number 0
  return size(comm) > 1 && rank(comm) == 0;
}
//-----------------------------------------------------------------------------
bool dolfin::MPI::is_receiver(const MPI_Comm comm)
{
  // Always receive on processors with numbers > 0
  return size(comm) > 1 && rank(comm) > 0;
}
//-----------------------------------------------------------------------------
void dolfin::MPI::barrier(const MPI_Comm comm)
{
#ifdef HAS_MPI
  MPI_Barrier(comm);
#endif
}
//-----------------------------------------------------------------------------
std::size_t dolfin::MPI::global_offset(const MPI_Comm comm,
                                       std::size_t range, bool exclusive)
{
#ifdef HAS_MPI
  // Compute inclusive or exclusive partial reduction
  std::size_t offset = 0;
  MPI_Scan(&range, &offset, 1, mpi_type<std::size_t>(), MPI_SUM, comm);
  if (exclusive)
    offset -= range;
  return offset;
#else
  return 0;
#endif
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::int64_t>
dolfin::MPI::local_range(const MPI_Comm comm, std::int64_t N)
{
  return local_range(comm, rank(comm), N);
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::int64_t>
dolfin::MPI::local_range(const MPI_Comm comm, int process, std::int64_t N)
{
  return compute_local_range(process, N, size(comm));
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::int64_t>
dolfin::MPI::compute_local_range(int process, std::int64_t N, int size)
{
  dolfin_assert(process >= 0);
  dolfin_assert(N >= 0);
  dolfin_assert(size > 0);

  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  if (process == 0)
  {
    std::cout << "n, r: " << n << ", " << r << std::endl;
    std::cout << "N, size: " << N << ", " << size << std::endl;

  }

  // Compute local range
  if (process < r)
    return {process*(n + 1), process*(n + 1) + n + 1};
  else
    return {process*n + r,  process*n + r + n};
}
//-----------------------------------------------------------------------------
unsigned int dolfin::MPI::index_owner(const MPI_Comm comm,
                                      std::size_t index, std::size_t N)
{
  dolfin_assert(index < N);

  // Get number of processes
  const unsigned int _size = size(comm);

  // Compute number of items per process and remainder
  const std::size_t n = N / _size;
  const std::size_t r = N % _size;

  // First r processes own n + 1 indices
  if (index < r * (n + 1))
    return index / (n + 1);

  // Remaining processes own n indices
  return r + (index - r * (n + 1)) / n;
}
//-----------------------------------------------------------------------------
#ifdef HAS_MPI
template<>
  dolfin::Table dolfin::MPI::all_reduce(const MPI_Comm comm,
                                        const dolfin::Table& table,
                                        const MPI_Op op)
{
  const std::string new_title = "[" + operation_map[op] + "] "
                              + table.name();

  // Handle trivial reduction
  if (MPI::size(comm) == 1)
  {
    Table table_all(table);
    table_all.rename(new_title, table_all.label());
    return table_all;
  }

  // Get keys, values into containers
  std::string keys;
  std::vector<double> values;
  keys.reserve(128*table.dvalues.size());
  values.reserve(table.dvalues.size());
  for (auto it = table.dvalues.begin(); it != table.dvalues.end(); ++it)
  {
    keys += it->first.first + '\0' + it->first.second + '\0';
    values.push_back(it->second);
  }

  // Gather to rank zero
  std::vector<std::string> keys_all;
  std::vector<double> values_all;
  gather(comm, keys, keys_all, 0);
  gather(comm, values, values_all, 0);

  // Return empty table on rank > 0
  if (MPI::rank(comm) > 0)
    return Table(new_title);

  // Prepare reduction operation y := op(y, x)
  void (*op_impl)(double&, const double&) = NULL;
  if (op == MPI_SUM || op == MPI_AVG())
    op_impl = [](double& y, const double& x){ y += x; };
  else if (op == MPI_MIN)
    op_impl = [](double& y, const double& x){ if (x<y) y = x; };
  else if (op == MPI_MAX)
    op_impl = [](double& y, const double& x){ if (x>y) y = x; };
  else
    dolfin_error("MPI.h",
                 "perform reduction of Table",
                 "MPI::reduce(comm, table, %d) not implemented",
                 op);

  // Construct dvalues map from obtained data
  std::map<std::pair<std::string, std::string>, double> dvalues_all;
  std::map<std::pair<std::string, std::string>, double>::iterator it;
  std::pair<std::string, std::string> key;
  key.first.reserve(128);
  key.second.reserve(128);
  double* values_ptr = values_all.data();
  for (unsigned int i = 0; i != MPI::size(comm); ++i)
  {
    std::stringstream keys_stream(keys_all[i]);
    while (std::getline(keys_stream, key.first, '\0'),
           std::getline(keys_stream, key.second, '\0'))
    {
      it = dvalues_all.find(key);
      if (it != dvalues_all.end())
        op_impl(it->second, *(values_ptr++));
      else
        dvalues_all[key] = *(values_ptr++);
    }
  }
  dolfin_assert(values_ptr == values_all.data() + values_all.size());

  // Weight by MPI size when averaging
  if (op == MPI_AVG())
  {
    const double w = 1.0 / static_cast<double>(size(comm));
    for (auto& it : dvalues_all)
      it.second *= w;
  }

  // Construct table to return
  Table table_all(new_title);
  for (auto& it : dvalues_all)
    table_all(it.first.first, it.first.second) = it.second;

  return table_all;
}
#endif
//-----------------------------------------------------------------------------
#ifdef HAS_MPI
template<>
  dolfin::Table dolfin::MPI::avg(MPI_Comm comm, const dolfin::Table& table)
{
  return all_reduce(comm, table, MPI_AVG());
}
#endif
//-----------------------------------------------------------------------------
#ifdef HAS_MPI
std::map<MPI_Op, std::string> dolfin::MPI::operation_map =
{
  { MPI_SUM, "MPI_SUM" },
  { MPI_MAX, "MPI_MAX" },
  { MPI_MIN, "MPI_MIN" }
};
#endif
//-----------------------------------------------------------------------------
#ifdef HAS_MPI
MPI_Op dolfin::MPI::MPI_AVG()
{
  // Return dummy MPI_Op which we identify with average
  static MPI_Op op = MPI_OP_NULL;
  static MPI_User_function* fn = [](void*, void*, int*, MPI_Datatype*){ };
  if (op == MPI_OP_NULL)
  {
    dolfin::SubSystemsManager::init_mpi();
    MPI_Op_create(fn, 1, &op);
    operation_map[op] = "MPI_AVG";
  }
  return op;
}
#endif
//-----------------------------------------------------------------------------
