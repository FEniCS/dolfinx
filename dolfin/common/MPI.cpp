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
#include <typeinfo>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/log/Table.h>
#include "SubSystemsManager.h"
#include "MPI.h"

namespace dolfin {

#ifdef HAS_MPI

//-----------------------------------------------------------------------------
MPIInfo::MPIInfo()
{
  MPI_Info_create(&info);
}
//-----------------------------------------------------------------------------
MPIInfo::~MPIInfo()
{
  MPI_Info_free(&info);
}
//-----------------------------------------------------------------------------
MPI_Info& MPIInfo::operator*()
{
  return info;
}
//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
unsigned int MPI::rank(const MPI_Comm comm)
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
unsigned int MPI::size(const MPI_Comm comm)
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
bool MPI::is_broadcaster(const MPI_Comm comm)
{
  // Always broadcast from processor number 0
  return size(comm) > 1 && rank(comm) == 0;
}
//-----------------------------------------------------------------------------
bool MPI::is_receiver(const MPI_Comm comm)
{
  // Always receive on processors with numbers > 0
  return size(comm) > 1 && rank(comm) > 0;
}
//-----------------------------------------------------------------------------
void MPI::barrier(const MPI_Comm comm)
{
#ifdef HAS_MPI
  MPI_Barrier(comm);
#endif
}
//-----------------------------------------------------------------------------
std::size_t MPI::global_offset(const MPI_Comm comm,
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
std::pair<std::size_t, std::size_t>
MPI::local_range(const MPI_Comm comm, std::size_t N)
{
  return local_range(comm, rank(comm), N);
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
MPI::local_range(const MPI_Comm comm, unsigned int process,
                         std::size_t N)
{
  return compute_local_range(process, N, size(comm));
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
MPI::compute_local_range(unsigned int process,
                                 std::size_t N,
                                 unsigned int size)
{
  // Compute number of items per process and remainder
  const std::size_t n = N / size;
  const std::size_t r = N % size;

  // Compute local range
  std::pair<std::size_t, std::size_t> range;
  if (process < r)
  {
    range.first = process*(n + 1);
    range.second = range.first + n + 1;
  }
  else
  {
    range.first = process*n + r;
    range.second = range.first + n;
  }

  return range;
}
//-----------------------------------------------------------------------------
unsigned int MPI::index_owner(const MPI_Comm comm,
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
  // Specialization for dolfin::log::Table class
  template<>
    Table MPI::all_reduce(const MPI_Comm comm, const Table& table, MPI_Op op)
  {
    #ifdef HAS_MPI
    // Get keys, values into containers
    std::string keys;
    std::vector<double> values;
    keys.reserve(128*table.dvalues.size());
    values.reserve(table.dvalues.size());
    for (auto it = table.dvalues.begin(); it != table.dvalues.end(); ++it)
    {
      keys += it->first.first + "\0" + it->first.second + "\0";
      values.push_back(it->second);
    }

    // Gather to rank zero
    std::vector<std::string> keys_all;
    std::vector<double> values_all;
    gather(comm, keys, keys_all, 0);
    gather(comm, values, values_all, 0);

    // Build the result
    if (MPI::rank(comm) == 0)
    {
      Table table_all(std::string("Reduced ") + typeid(op).name()
                      + ": " + table.title());
      std::string key0, key1;
      key0.reserve(128);
      key1.reserve(128);
      double* values_ptr = values_all.data();
      for (unsigned int i = 0; i != MPI::size(comm); ++i)
      {
        std::stringstream keys_stream(keys_all[i]);
        while (std::getline(keys_stream, key0, '\0'),
               std::getline(keys_stream, key1, '\0'))
        {
          // FIXME: What is unset value?
          const double value = table_all.get_value(key0, key1);
          if (op == MPI_SUM)
            table_all(key0, key1) = value + *(values_ptr++);
          else if (op == MPI_MIN)
            table_all(key0, key1) = std::min(value, *(values_ptr++));
          else if (op == MPI_MAX)
            table_all(key0, key1) = std::max(value, *(values_ptr++));
          else
            dolfin_error("MPI.h",
                         "perform reduction of Table",
                         "MPI::reduce(comm, table, %s) not implemented",
                         typeid(op).name());
        }
      }
      return table_all;
    }
    else
      return Table();
    #else
    return value;
    #endif
  }
  //---------------------------------------------------------------------------
}
