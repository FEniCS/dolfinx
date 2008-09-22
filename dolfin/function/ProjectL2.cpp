// Copyright (C) 2008 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-03-18
// Last changed: 2008-03-18
//

#include <dolfin/elements/ProjectionLibrary.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/pde/LinearPDE.h>
#include "Function.h"
#include "ProjectL2.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::projectL2(Mesh& meshB, Function& fA, Function& fB, FiniteElement& element)
{
  error("dolfin::projectL2 is disabled. Will be fixed when new Function is in place.");
  /*
  Form* a = ProjectionLibrary::create_projection_a(element.signature());
  Form* L = ProjectionLibrary::create_projection_L(element.signature(), fA);

  // Compute projection
  // FIXME: LinearPDE should not own memory from fB, allocate on heap for now
  LinearPDE* pde = new LinearPDE(*a, *L, meshB);
  pde->solve(fB);
  */
}
//-----------------------------------------------------------------------------
void dolfin::projectL2NonMatching(Mesh& meshB, Function& fA, Function& fB,
				  FiniteElement& element)
{
  error("dolfin::projectL2 is disabled. Will be fixed when new Function is in place.");
  /*
  // Create non-matching function fN
  NonMatchingFunction fN(meshB, fA);

  Form* a = ProjectionLibrary::create_projection_a(element.signature());
  Form* L = ProjectionLibrary::create_projection_L(element.signature(), fN);

  // Compute projection
  // FIXME: LinearPDE should not own memory from fB, allocate on heap for now
  LinearPDE* pde = new LinearPDE(*a, *L, meshB);
  pde->solve(fB);
  */
}
//-----------------------------------------------------------------------------
NonMatchingFunction::NonMatchingFunction(Mesh& mesh, Function& fA) :
  Function(mesh), fA(fA)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonMatchingFunction::eval(real* values, const real* x) const
{
  // Evaluate discrete function fA pointwise
  fA.eval(values, x);
}
//-----------------------------------------------------------------------------
