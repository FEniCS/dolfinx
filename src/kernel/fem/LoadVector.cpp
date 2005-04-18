// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/LoadVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LoadVector::LoadVector(Mesh& mesh) : Vector(mesh.noNodes())
{
  // FIXME: BROKEN
  dolfin_error("Not implemented for new system.");
}
//-----------------------------------------------------------------------------
