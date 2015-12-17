// Copyright (C) 2012 Garth N. Wells
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

#include <memory>

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include "IndexMap.h"
#include "SparsityPattern.h"
#include "TensorLayout.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
TensorLayout::TensorLayout(std::size_t pdim, Sparsity sparsity_pattern)
  : primary_dim(pdim), _mpi_comm(MPI_COMM_NULL)
{
  // Create empty sparsity pattern
  if (sparsity_pattern == TensorLayout::Sparsity::SPARSE)
    _sparsity_pattern = std::make_shared<SparsityPattern>(primary_dim);
}
//-----------------------------------------------------------------------------
TensorLayout::TensorLayout(const MPI_Comm mpi_comm,
             std::vector<std::shared_ptr<const IndexMap>> index_maps,
             std::size_t pdim,
             Sparsity sparsity_pattern,
             Ghosts ghosted)
  : primary_dim(pdim), _mpi_comm(mpi_comm), _index_maps(index_maps),
    _ghosted(ghosted)
{
  if (sparsity_pattern == TensorLayout::Sparsity::SPARSE)
    _sparsity_pattern = std::make_shared<SparsityPattern>(primary_dim);

  // Only rank 2 sparsity patterns are supported
  dolfin_assert(!(_sparsity_pattern && index_maps.size() != 2));
}
//-----------------------------------------------------------------------------
void TensorLayout::init(
  const MPI_Comm mpi_comm,
  const std::vector<std::shared_ptr<const IndexMap>> index_maps,
  const Ghosts ghosted)
{
  // Only rank 2 sparsity patterns are supported
  dolfin_assert(!(_sparsity_pattern && index_maps.size() != 2));

  // Store everything
  _index_maps = index_maps;
  _mpi_comm = mpi_comm;
  _ghosted = ghosted;
}
//-----------------------------------------------------------------------------
std::size_t TensorLayout::rank() const
{
  return _index_maps.size();
}
//-----------------------------------------------------------------------------
std::size_t TensorLayout::size(std::size_t i) const
{
  dolfin_assert(i < _index_maps.size());
  return _index_maps[i]->size(IndexMap::MapSize::GLOBAL);
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
TensorLayout::local_range(std::size_t dim) const
{
  dolfin_assert(dim < _index_maps.size());
  return _index_maps[dim]->local_range();

}
//-----------------------------------------------------------------------------
std::string TensorLayout::str(bool verbose) const
{
  std::stringstream s;
  s << "<TensorLayout for tensor of rank " << rank() << ">" << std::endl;
  for (std::size_t i = 0; i < rank(); i++)
  {
    s << " Local range for dim " << i << ": [" << local_range(i).first
      << ", " << local_range(i).second << ")" << std::endl;
  }
  return s.str();
}
//-----------------------------------------------------------------------------
