// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-16
// Last changed: 2009-04-01

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/FiniteElement.h>
#include "FunctionPlotData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionPlotData::FunctionPlotData(const Function& v)
  : Variable(v.name(), v.label())
{
  // Copy mesh
  mesh = v.function_space().mesh();

  // Compute number of entries in tensor value (entries per vertex)
  uint size = 1;
  for (uint i = 0; i < v.function_space().element().value_rank(); i++)
    size *= v.function_space().element().value_dimension(i);

  // Initialize vector
  const uint N = size*mesh.num_vertices();
  vertex_values.resize(N);

  // Interpolate vertex values
  double* values = new double[N];
  v.interpolate(values);
  vertex_values.set(values);
  delete [] values;

  // Get shape and dimension
  rank = v.function_space().element().value_rank();
  if (rank > 1)
    error("Plotting of rank %d functions not supported.", rank);
}
//-----------------------------------------------------------------------------
FunctionPlotData::FunctionPlotData() : rank(0)
{
  // Do nothing  
}
//-----------------------------------------------------------------------------
FunctionPlotData::~FunctionPlotData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
