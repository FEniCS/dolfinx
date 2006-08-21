// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-08-21
// Last changed: 2006-08-21

#include <dolfin/GenericMatrix.h>
#include <dolfin/GenericVector.h>
#include <dolfin/Mesh.h>
#include <dolfin/FEM.h>
#include <dolfin/MatrixFactory.h>

#include "ffc-forms/MassMatrix2D.h"
#include "ffc-forms/MassMatrix3D.h"
#include "ffc-forms/StiffnessMatrix2D.h"
#include "ffc-forms/StiffnessMatrix3D.h"
#include "ffc-forms/ConvectionMatrix2D.h"
#include "ffc-forms/ConvectionMatrix3D.h"
#include "ffc-forms/LoadVector2D.h"
#include "ffc-forms/LoadVector3D.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MatrixFactory::computeMassMatrix(GenericMatrix& A, Mesh& mesh)
{
  if ( mesh.type() == Mesh::triangles )
  {
    MassMatrix2D::BilinearForm a;
    FEM::assemble(a, A, mesh);
  }
  else if ( mesh.type() == Mesh::tetrahedra )
  {
    MassMatrix3D::BilinearForm a;
    FEM::assemble(a, A, mesh);
  }
  else
  {
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeStiffnessMatrix(GenericMatrix& A, Mesh& mesh,
					   real c)
{
  if ( mesh.type() == Mesh::triangles )
  {
    StiffnessMatrix2D::BilinearForm a(c);
    FEM::assemble(a, A, mesh);
  }
  else if ( mesh.type() == Mesh::tetrahedra )
  {
    StiffnessMatrix3D::BilinearForm a(c);
    FEM::assemble(a, A, mesh);
  }
  else
  {
    dolfin_error("Unknown mesh type.");
  } 
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeConvectionMatrix(GenericMatrix& A, Mesh& mesh,
					    real cx, real cy, real cz)
{
  if ( mesh.type() == Mesh::triangles )
  {
    ConvectionMatrix2D::BilinearForm a(cx, cy);
    FEM::assemble(a, A, mesh);
  }
  else if ( mesh.type() == Mesh::tetrahedra )
  {
    ConvectionMatrix3D::BilinearForm a(cx, cy, cz);
    FEM::assemble(a, A, mesh);
  }
  else
  {
    dolfin_error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeLoadVector(GenericVector& x, Mesh& mesh, real c)
{
  if ( mesh.type() == Mesh::triangles )
  {
    LoadVector2D::LinearForm b(c);
    FEM::assemble(b, x, mesh);
  }
  else if ( mesh.type() == Mesh::tetrahedra )
  {
    LoadVector3D::LinearForm b(c);
    FEM::assemble(b, x, mesh);
  }
  else
  {
    dolfin_error("Unknown mesh type.");
  } 
}
//-----------------------------------------------------------------------------
