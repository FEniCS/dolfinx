// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-09-11

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
