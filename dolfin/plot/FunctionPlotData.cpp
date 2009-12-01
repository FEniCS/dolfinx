// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-16
// Last changed: 2009-10-07

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
  double* values = new double[N];
  v.compute_vertex_values(values, mesh);
  _vertex_values->set_local(values);
  delete [] values;
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
