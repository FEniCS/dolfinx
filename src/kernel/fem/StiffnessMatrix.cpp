// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/FEM.h>
#include <dolfin/Mesh.h>
#include <dolfin/StiffnessMatrix2D.h>
#include <dolfin/StiffnessMatrix3D.h>
#include <dolfin/StiffnessMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
StiffnessMatrix::StiffnessMatrix(Mesh& mesh, real epsilon)
  : Matrix(mesh.noNodes(), mesh.noNodes())
{
  if ( mesh.type() == Mesh::triangles )
  {
    StiffnessMatrix2D::BilinearForm a;
    FEM::assemble(a, *this, mesh);
  }
  else if ( mesh.type() == Mesh::tetrahedrons )
  {
    StiffnessMatrix3D::BilinearForm a;
    FEM::assemble(a, *this, mesh);
  }
  else
  {
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
