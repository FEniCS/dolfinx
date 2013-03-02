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
//
// First added:  2012-02-24
// Last changed:

#include <dolfin/log/log.h>
#include <dolfin/log/LogStream.h>
#include "SparsityPattern.h"
#include "TensorLayout.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
TensorLayout::TensorLayout(std::size_t pdim, bool sparsity_pattern)
    : primary_dim(pdim)
{
  // Create empty sparsity pattern
  if (sparsity_pattern)
    _sparsity_pattern.reset(new SparsityPattern(primary_dim));
}
//-----------------------------------------------------------------------------
TensorLayout::TensorLayout(const std::vector<std::size_t>& dims, std::size_t pdim,
  const std::vector<std::pair<std::size_t, std::size_t> >& ownership_range,
  bool sparsity_pattern)
  : primary_dim(pdim), _shape(dims), _ownership_range(ownership_range)
{
  // Only rank 2 sparsity patterns are supported
  dolfin_assert(!(sparsity_pattern && dims.size() != 2));

  // Check that dimensions match
  dolfin_assert(dims.size() == ownership_range.size());

  // Create empty sparsity pattern
  if (_sparsity_pattern)
    _sparsity_pattern.reset(new SparsityPattern(primary_dim));
}
//-----------------------------------------------------------------------------
void TensorLayout::init(const std::vector<std::size_t>& dims,
  const std::vector<std::pair<std::size_t, std::size_t> >& ownership_range)
{
  // Only rank 2 sparsity patterns are supported
  dolfin_assert(!(_sparsity_pattern && dims.size() != 2));

  // Check that dimensions match
  dolfin_assert(dims.size() == ownership_range.size());

  // Store dimensions
  _shape = dims;

  // Store ownership range
  _ownership_range = ownership_range;
}
//-----------------------------------------------------------------------------
std::size_t TensorLayout::rank() const
{
  return _shape.size();
}
//-----------------------------------------------------------------------------
std::size_t TensorLayout::size(std::size_t i) const
{
  dolfin_assert(i < _shape.size());
  return _shape[i];
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> TensorLayout::local_range(std::size_t dim) const
{
  dolfin_assert(dim < 2);
  return _ownership_range[dim];
}
//-----------------------------------------------------------------------------
std::string TensorLayout::str() const
{
  std::stringstream s;
  s << "<TensorLayout for tensor of rank " << rank() << ">" << std::endl;
  for (std::size_t i = 0; i < rank(); i++)
  {
    s << " Local range for dim " << i << ": [" << _ownership_range[i].first
        << ", " << _ownership_range[i].second << ")" << std::endl;
  }
  return s.str();
}
//-----------------------------------------------------------------------------
