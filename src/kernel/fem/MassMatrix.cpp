// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-03-31
// Last changed: 2006-05-07

#include <dolfin/FEM.h>
#include <dolfin/Mesh.h>
#include <dolfin/MassMatrix2D.h>
#include <dolfin/MassMatrix3D.h>
#include <dolfin/MassMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MassMatrix::MassMatrix(Mesh& mesh) : Matrix(mesh.numVertices(), mesh.numVertices())
{
  if ( mesh.type() == Mesh::triangles )
  {
    MassMatrix2D::BilinearForm a;
    FEM::assemble(a, *this, mesh);
  }
  else if ( mesh.type() == Mesh::tetrahedra )
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
