// Copyright (C) 2008-2009 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2009-02-25

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/FiniteElement.h>
#include "TransfiniteInterpolation.h"
#include "HarmonicSmoothing.h"
#include "ALE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh, Mesh& new_boundary, ALEType method)
{
  switch (method)
  {
  case lagrange:
    cout << "Updating mesh coordinates using transfinite mean value interpolation (Lagrange)." << endl;
    TransfiniteInterpolation::move(mesh, new_boundary,
                                   TransfiniteInterpolation::interpolation_lagrange);
    break;
  case hermite:
    cout << "Updating mesh coordinates using transfinite mean value interpolation (Hermite)." << endl;
    TransfiniteInterpolation::move(mesh, new_boundary,
                                   TransfiniteInterpolation::interpolation_hermite);
    break;
  case harmonic:
    cout << "Updating mesh coordinates using harmonic smoothing." << endl;
    HarmonicSmoothing::move(mesh, new_boundary);
    break;
  default:
    error("Unknown method for ALE mesh smoothing.");
  }
}
//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh, const Function& displacement)
{
  // Check dimensions
  const FiniteElement& element = displacement.function_space().element();
  const uint gdim = mesh.geometry().dim();
  if (!((element.value_rank() == 0 && gdim == 0) ||
        (element.value_rank() == 1 && gdim == element.value_dimension(0))))
    error("Unable to move mesh, illegal value dimension of displacement function.");

  // Interpolate at vertices
  const uint N = mesh.numVertices();
  double* vertex_values = new real[N*gdim];
  displacement.interpolate(vertex_values);

  // Move vertex coordinates
  double* x = mesh.geometry().x();
  for (uint d = 0; d < gdim; d++)
  {
    for (uint i = 0; i < N; i++)
      x[i*gdim + d] += vertex_values[d*N + i];
  }

  // Clean up
  delete [] vertex_values;
}
//-----------------------------------------------------------------------------
