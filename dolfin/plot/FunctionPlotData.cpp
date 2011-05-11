// Copyright (C) 2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-16
// Last changed: 2009-10-07

#include <dolfin/common/Array.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/DefaultFactory.h>
#include "FunctionPlotData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionPlotData::FunctionPlotData(const GenericFunction& v, const Mesh& mesh)
  : Variable(v.name(), v.label())
{
  // Copy the mesh (yes, *copy*)
  this->mesh = mesh;

  // Check and store rank
  rank = v.value_rank();
  if (rank > 1)
    error("Plotting of rank %d functions not supported.", rank);

  // Initialize local vector
  DefaultFactory factory;
  _vertex_values.reset(factory.create_local_vector());
  assert(_vertex_values);
  const uint N = v.value_size()*mesh.num_vertices();
  _vertex_values->resize(N);

  // Compute vertex values
  Array<double> values(N);
  v.compute_vertex_values(values, mesh);
  _vertex_values->set_local(values);
}
//-----------------------------------------------------------------------------
FunctionPlotData::FunctionPlotData() : rank(0)
{
  DefaultFactory factory;
  _vertex_values.reset(factory.create_local_vector());
  assert(_vertex_values);
}
//-----------------------------------------------------------------------------
FunctionPlotData::~FunctionPlotData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericVector& FunctionPlotData::vertex_values() const
{
  assert(_vertex_values);
  return *_vertex_values;
}
//-----------------------------------------------------------------------------
