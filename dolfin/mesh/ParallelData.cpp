// Copyright (C) 2011 Anders Logg and Garth N. Wells
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
// First added:  2011-01-17
// Last changed: 2012-10-01

#include "MeshFunction.h"
#include "ParallelData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ParallelData::ParallelData(const Mesh& mesh)
  : _exterior_facet(new MeshFunction<bool>(mesh))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ParallelData::ParallelData(const ParallelData& data)
  : _exterior_facet(new MeshFunction<bool>(*data._exterior_facet))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ParallelData::~ParallelData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshFunction<bool>& ParallelData::exterior_facet()
{
  dolfin_assert(_exterior_facet);
  return *_exterior_facet;
}
//-----------------------------------------------------------------------------
const MeshFunction<bool>& ParallelData::exterior_facet() const
{
  dolfin_assert(_exterior_facet);
  return *_exterior_facet;
}
//-----------------------------------------------------------------------------
