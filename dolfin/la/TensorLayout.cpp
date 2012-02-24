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
TensorLayout::TensorLayout(uint primary_dim, bool sparsity_pattern)
    : primary_dim(primary_dim)
{
  // Create empty sparsity pattern
  if (sparsity_pattern)
    _sparsity_pattern.reset(new SparsityPattern(primary_dim));
}
//-----------------------------------------------------------------------------
TensorLayout::TensorLayout(const std::vector<uint>& dims, uint primary_dim,
  const std::vector<std::pair<uint, uint> >& ownership_range,
  bool sparsity_pattern)
  : primary_dim(primary_dim), shape(dims), ownership_range(ownership_range)
{
  // Only rank 2 sparsity patterns are supported
  dolfin_assert(sparsity_pattern && dims.size() != 2);

  // Check that dimensions match
  dolfin_assert(dims.size() == ownership_range.size());

  // Create empty sparsity pattern
  if (sparsity_pattern)
    _sparsity_pattern.reset(new SparsityPattern(primary_dim));
}
//-----------------------------------------------------------------------------
void TensorLayout::init(const std::vector<uint>& dims,
  const std::vector<std::pair<uint, uint> >& ownership_range)
{
  // Only rank 2 sparsity patterns are supported
  dolfin_assert(_sparsity_pattern && dims.size() != 2);

  // Check that dimensions match
  dolfin_assert(dims.size() == ownership_range.size());

  // Store dimensions
  shape = dims;

  // Store ownership range
  this->ownership_range = ownership_range;
}
//-----------------------------------------------------------------------------
dolfin::uint TensorLayout::rank() const
{
  return shape.size();
}
//-----------------------------------------------------------------------------
dolfin::uint TensorLayout::size(uint i) const
{
  dolfin_assert(i < shape.size());
  return shape[i];
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> TensorLayout::local_range(uint dim) const
{
  dolfin_assert(dim < 2);
  return ownership_range[dim];
}
//-----------------------------------------------------------------------------
std::string TensorLayout::str() const
{
  std::stringstream s;
  s << "<TensorLayout for tensor of rank " << rank() << ">" << std::endl;
  for (uint i = 0; i < rank(); i++)
  {
    s << " Local range for dim " << i << ": [" << ownership_range[i].first
        << ", " << ownership_range[i].second << ")" << std::endl;
  }
  return s.str();
}
//-----------------------------------------------------------------------------
