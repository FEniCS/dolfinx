// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-16
// Last changed: 2009-03-16

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/FiniteElement.h>
#include "FunctionPlotData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionPlotData::FunctionPlotData(const Function& v)
{
  // Copy mesh
  mesh = v.function_space().mesh();

  // Initialize vector
  const uint N = v.function_space().dim();
  vertex_values.resize(N);

  // Interpolate vertex values
  double* values = new double[N];
  v.interpolate(values);
  vertex_values.set(values);
  delete [] values;

  // Get shape and dimension
  for (uint i = 0; i < v.function_space().element().value_rank(); i++)
    value_shape.push_back(v.function_space().element().value_dimension(i));
}
//-----------------------------------------------------------------------------
FunctionPlotData::~FunctionPlotData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
