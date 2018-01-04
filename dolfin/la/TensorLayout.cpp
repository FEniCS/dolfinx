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
TensorLayout::TensorLayout(MPI_Comm comm, std::size_t pdim,
                           Sparsity sparsity_pattern)
  : primary_dim(pdim), _mpi_comm(comm)
{
  // Create empty sparsity pattern
  if (sparsity_pattern == TensorLayout::Sparsity::SPARSE)
    _sparsity_pattern = std::make_shared<SparsityPattern>(comm, primary_dim);
}
//-----------------------------------------------------------------------------
TensorLayout::TensorLayout(MPI_Comm comm,
                           std::array<std::shared_ptr<const IndexMap>, 2> index_maps,
                           std::size_t pdim,
                           Sparsity sparsity_pattern,
                           Ghosts ghosted)
  : primary_dim(pdim), _mpi_comm(comm), _index_maps(index_maps),
    _ghosted(ghosted)
{
  if (sparsity_pattern == TensorLayout::Sparsity::SPARSE)
    _sparsity_pattern = std::make_shared<SparsityPattern>(comm, primary_dim);
}
//-----------------------------------------------------------------------------
void TensorLayout::init(const std::array<std::shared_ptr<const IndexMap>, 2> index_maps,
                        const Ghosts ghosted)
{
  // Store everything
  _index_maps = index_maps;
  _ghosted = ghosted;
}
//-----------------------------------------------------------------------------
std::size_t TensorLayout::size(std::size_t i) const
{
  dolfin_assert(i < _index_maps.size());
  return _index_maps[i]->block_size()*_index_maps[i]->size(IndexMap::MapSize::GLOBAL);
}
//-----------------------------------------------------------------------------
std::array<std::size_t, 2> TensorLayout::local_range(std::size_t dim) const
{
  dolfin_assert(dim < _index_maps.size());
  std::size_t bs = _index_maps[dim]->block_size();
  auto lrange = _index_maps[dim]->local_range();
  return {bs*lrange[0], bs*lrange[1]};
}
//-----------------------------------------------------------------------------
std::string TensorLayout::str(bool verbose) const
{
  std::stringstream s;
  s << "<TensorLayout for matrix>" << std::endl;
  for (std::size_t i = 0; i < 2; i++)
  {
    s << " Local range for dim " << i << ": [" << local_range(i)[0]
      << ", " << local_range(i)[1] << ")" << std::endl;
  }
  return s.str();
}
//-----------------------------------------------------------------------------
