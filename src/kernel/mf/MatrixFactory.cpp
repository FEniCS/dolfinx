// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-21
// Last changed: 2006-10-26

#include <dolfin/assemble.h>
#include <dolfin/GenericMatrix.h>
#include <dolfin/GenericVector.h>
#include <dolfin/Mesh.h>
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
  if (mesh.type().cellType() == CellType::triangle)
  {
    MassMatrix2DBilinearForm a;
    assemble(A, a, mesh);
  }
  else if (mesh.type().cellType() == CellType::tetrahedron)
  {
    MassMatrix3DBilinearForm a;
    assemble(A, a, mesh);
  }
  else
  {
    error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeStiffnessMatrix(GenericMatrix& A, Mesh& mesh,
					   real c)
{
  Function f(mesh, c);

  if (mesh.type().cellType() == CellType::triangle)
  {
    StiffnessMatrix2DBilinearForm a(f);
    assemble(A, a, mesh);
  }
  else if (mesh.type().cellType() == CellType::tetrahedron)
  {
    StiffnessMatrix3DBilinearForm a(f);
    assemble(A, a, mesh);
  }
  else
  {
    error("Unknown mesh type.");
  } 
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeConvectionMatrix(GenericMatrix& A, Mesh& mesh,
					    real cx, real cy, real cz)
{
  Function fx(mesh, cx);
  Function fy(mesh, cy);
  Function fz(mesh, cz);

  if (mesh.type().cellType() == CellType::triangle)
  {
    ConvectionMatrix2DBilinearForm a(fx, fy);
    assemble(A, a, mesh);
  }
  else if (mesh.type().cellType() == CellType::tetrahedron)
  {
    ConvectionMatrix3DBilinearForm a(fx, fy, fz);
    assemble(A, a, mesh);
  }
  else
  {
    error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeLoadVector(GenericVector& x, Mesh& mesh, real c)
{
  Function f(mesh, c);

  error("MF forms need to be updated to new mesh format.");
  if (mesh.type().cellType() == CellType::triangle)
  {
    LoadVector2DLinearForm b(f);
    assemble(x, b, mesh);
  }
  else if (mesh.type().cellType() == CellType::tetrahedron)
  {
    LoadVector3DLinearForm b(f);
    assemble(x, b, mesh);
  }
  else
  {
    error("Unknown mesh type.");
  } 
}
//-----------------------------------------------------------------------------
