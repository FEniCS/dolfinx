// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring
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
// Modified by Anders Logg, 2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2008-04-21
// Last changed: 2009-08-06

#ifdef HAS_TRILINOS

#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <Epetra_FECrsGraph.h>

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include "EpetraFactory.h"
#include "GenericSparsityPattern.h"
#include "EpetraSparsityPattern.h"


using namespace dolfin;
using dolfin::uint;

//-----------------------------------------------------------------------------
EpetraSparsityPattern::EpetraSparsityPattern() : _rank(0), epetra_graph(0)
{
  dims[0] = 0;
  dims[1] = 0;
}
//-----------------------------------------------------------------------------
EpetraSparsityPattern::~EpetraSparsityPattern()
{
  delete epetra_graph;
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::init(uint rank_, const uint* dims_)
{
  _rank = rank_;

  if (_rank == 1)
    dims[0] = dims_[0];
  else if (_rank == 2)
  {
    dims[0] = dims_[0];
    dims[1] = dims_[1];

    const std::pair<uint, uint> range = MPI::local_range(dims[0]);
    const uint num_local_rows = range.second - range.first;

    EpetraFactory& f = EpetraFactory::instance();
    Epetra_MpiComm comm = f.get_mpi_comm();
    Epetra_Map row_map(dims[0], num_local_rows, 0, comm);
    epetra_graph = new Epetra_FECrsGraph(Copy, row_map, 0);
  }
  else
  {
    dolfin_error("EpetraSparsityPattern.cpp",
                 "initialize Epetra sparsity pattern",
                 "illegal rank (%d)", _rank);
  }
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::insert(const uint* num_rows,
                                   const uint * const * rows)
{
  if (_rank == 2)
  {
    epetra_graph->InsertGlobalIndices(num_rows[0], reinterpret_cast<const int*>(rows[0]),
                                      num_rows[1], reinterpret_cast<const int*>(rows[1]));
  }
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraSparsityPattern::rank() const
{
  return _rank;
}
//-----------------------------------------------------------------------------
uint EpetraSparsityPattern::size(uint i) const
{
  if (_rank == 1)
    return dims[0];

  if (_rank == 2)
  {
    assert(epetra_graph);
    if (i == 0)
      return epetra_graph->NumGlobalRows();
    else
      return epetra_graph->NumGlobalCols();
  }
  return 0;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> EpetraSparsityPattern::local_range(uint dim) const
{
  dolfin_error("EpetraSparsityPattern.cpp",
               "access local range for Epetra sparsity pattern",
               "not implemented");
  return std::make_pair(0, 0);
  //assert(dim < 2);
  //return MPI::local_range(size(dim));
}
//-----------------------------------------------------------------------------
uint EpetraSparsityPattern::num_nonzeros() const
{
  dolfin_not_implemented();
  return 0;
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::num_nonzeros_diagonal(std::vector<uint>& num_nonzeros) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::num_nonzeros_off_diagonal(std::vector<uint>& num_nonzeros) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
std::vector<std::vector<uint> > EpetraSparsityPattern::diagonal_pattern(Type type) const
{
  dolfin_not_implemented();
  return std::vector<std::vector<uint> >(0);
}
//-----------------------------------------------------------------------------
std::vector<std::vector<uint> > EpetraSparsityPattern::off_diagonal_pattern(Type type) const
{
  dolfin_not_implemented();
  return std::vector<std::vector<uint> >(0);
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::apply()
{
  assert(epetra_graph);
  epetra_graph->FillComplete();
  //epetra_graph->GlobalAssemble();
}
//-----------------------------------------------------------------------------
Epetra_FECrsGraph& EpetraSparsityPattern::pattern() const
{
  return *epetra_graph;
}
//-----------------------------------------------------------------------------

#endif
