// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/StiffnessMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
StiffnessMatrix::StiffnessMatrix(Mesh& mesh, real epsilon)
  : Matrix(mesh.noNodes(), mesh.noNodes())
{
  // FIXME: BROKEN
  dolfin_error("Not implemented for new system.");
}
//-----------------------------------------------------------------------------
