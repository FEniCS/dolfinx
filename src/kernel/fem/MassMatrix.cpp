// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/FEM.h>
#include <dolfin/Mesh.h>
#include <dolfin/MassMatrix2D.h>
#include <dolfin/MassMatrix3D.h>
#include <dolfin/MassMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MassMatrix::MassMatrix(Mesh& mesh) : Matrix(mesh.noNodes(), mesh.noNodes())
{
  if ( mesh.type() == Mesh::triangles )
  {
    MassMatrix2D::BilinearForm a;
    FEM::assemble(a, *this, mesh);
  }
  else if ( mesh.type() == Mesh::tetrahedrons )
  {
    MassMatrix3D::BilinearForm a;
    FEM::assemble(a, *this, mesh);
  }
  else
  {
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
