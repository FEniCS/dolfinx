// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-03-31
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/FEM.h>
#include <dolfin/Mesh.h>
#include <dolfin/StiffnessMatrix2D.h>
#include <dolfin/StiffnessMatrix3D.h>
#include <dolfin/StiffnessMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
StiffnessMatrix::StiffnessMatrix(Mesh& mesh, real epsilon)
  : Matrix(mesh.numVertices(), mesh.numVertices())
{
  if ( mesh.type() == Mesh::triangles )
  {
    StiffnessMatrix2D::BilinearForm a(epsilon);
    FEM::assemble(a, *this, mesh);
  }
  else if ( mesh.type() == Mesh::tetrahedra )
  {
    StiffnessMatrix3D::BilinearForm a(epsilon);
    FEM::assemble(a, *this, mesh);
  }
  else
  {
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------

#endif
