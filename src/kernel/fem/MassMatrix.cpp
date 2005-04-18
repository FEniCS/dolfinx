// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/MassMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MassMatrix::MassMatrix(Mesh& mesh) : Matrix(mesh.noNodes(), mesh.noNodes())
{
  // FIXME: BROKEN
  dolfin_error("Not implemented for new system.");
}
//-----------------------------------------------------------------------------
