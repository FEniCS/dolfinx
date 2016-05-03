// Copyright (C) 2014-2015 Anders Logg
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
// First added:  2014-06-11
// Last changed: 2015-11-12

#include <memory>
#include <dolfin/common/NoDeleter.h>
#include "FunctionSpace.h"
#include "MultiMeshSubSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshSubSpace::MultiMeshSubSpace(MultiMeshFunctionSpace& V,
                                     std::size_t component)
  : MultiMeshFunctionSpace(V.multimesh())
{
  // Create array
  std::vector<std::size_t> c = {component};

  // Build subspace
  _build(V, c);
}
//-----------------------------------------------------------------------------
MultiMeshSubSpace::MultiMeshSubSpace(MultiMeshFunctionSpace& V,
                                     std::size_t component,
                                     std::size_t sub_component)
  : MultiMeshFunctionSpace(V.multimesh())
{
  // Create array
  std::vector<std::size_t> c = {{component, sub_component}};

  // Build subspace
  _build(V, c);
}
//-----------------------------------------------------------------------------
MultiMeshSubSpace::MultiMeshSubSpace(MultiMeshFunctionSpace& V,
                                     const std::vector<std::size_t>& component)
  : MultiMeshFunctionSpace(V.multimesh())
{
  // Build subspace
  _build(V, component);
}
//-----------------------------------------------------------------------------
void MultiMeshSubSpace::_build(MultiMeshFunctionSpace& V,
                               const std::vector<std::size_t>& component)
{
  // Vector of offsets. Note that offsets need to be computed here and
  // not inside the dofmap builder since it does not know about the
  // offsets of the subdofmaps relative to the multimesh function
  // space on all parts.
  std::vector<dolfin::la_index> offsets = {0};

  // Extract proper subspaces for each part and add
  for (std::size_t part = 0; part < V.num_parts(); part++)
  {
    // Extract function space
    std::shared_ptr<const FunctionSpace> part_space(V.part(part));

    // Extract subspace
    auto part_subspace = part_space->sub(component);

    // Add the subspace
    this->add(part_subspace);

    // Add to offsets
    offsets.push_back(offsets[part] + part_space->dim());
  }

  // Build multimesh function space from subspaces
  this->build(offsets);
}
//-----------------------------------------------------------------------------
