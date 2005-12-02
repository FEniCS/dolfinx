// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-03-31
// Last changed: 2005-12-01

#include <dolfin/Mesh.h>
#include <dolfin/LoadVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LoadVector::LoadVector(Mesh& mesh) : Vector(mesh.noVertices())
{
  // FIXME: BROKEN
  dolfin_error("Not implemented for new system.");
}
//-----------------------------------------------------------------------------
