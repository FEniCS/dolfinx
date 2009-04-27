// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-21
// Last changed: 2006-10-26

#include <dolfin/fem/Assembler.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>
#include "MatrixFactory.h"

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
  warning("Using default dof map in MatrixFactory");
  if (mesh.type().cell_type() == CellType::triangle)
  {
    MassMatrix2DFunctionSpace V(mesh);
    MassMatrix2DBilinearForm a(V, V);
    Assembler::assemble(A, a);
  }
  else if (mesh.type().cell_type() == CellType::tetrahedron)
  {
    MassMatrix3DFunctionSpace V(mesh);
    MassMatrix3DBilinearForm a(V, V);
    Assembler::assemble(A, a);
  }
  else
    error("Unknown mesh type.");
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeStiffnessMatrix(GenericMatrix& A, Mesh& mesh, double c)
{
  // Create constant
  Constant f(c);

  warning("Using default dof map in MatrixFactory");
  if (mesh.type().cell_type() == CellType::triangle)
  {
    StiffnessMatrix2DFunctionSpace V(mesh);
    StiffnessMatrix2DBilinearForm a(V, V);
    a.c = f;
    Assembler::assemble(A, a);
  }
  else if (mesh.type().cell_type() == CellType::tetrahedron)
  {
    StiffnessMatrix3DFunctionSpace V(mesh);
    StiffnessMatrix3DBilinearForm a(V, V);
    a.c = f;
    Assembler::assemble(A, a);
  }
  else
    error("Unknown mesh type.");

}
//-----------------------------------------------------------------------------
void MatrixFactory::computeConvectionMatrix(GenericMatrix& A, Mesh& mesh,
					    double cx, double cy, double cz)
{
  error("MatrixFactory need to be updated for new Function interface.");
/*
  Function fx(mesh, cx);
  Function fy(mesh, cy);
  Function fz(mesh, cz);

  warning("Using default dof map in MatrixFactory");
  if (mesh.type().cell_type() == CellType::triangle)
  {
    ConvectionMatrix2DBilinearForm a(fx, fy);
    assemble(A, a, mesh);
  }
  else if (mesh.type().cell_type() == CellType::tetrahedron)
  {
    ConvectionMatrix3DBilinearForm a(fx, fy, fz);
    assemble(A, a, mesh);
  }
  else
  {
    error("Unknown mesh type.");
  }
*/
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeLoadVector(GenericVector& x, Mesh& mesh, double c)
{
  error("MatrixFactory need to be updated for new Function interface.");

/*
  Function f(mesh, c);

  error("MF forms need to be updated to new mesh format.");
  warning("Using default dof map in MatrixFactory");
  if (mesh.type().cell_type() == CellType::triangle)
  {
    LoadVector2DLinearForm b(f);
    assemble(x, b, mesh);
  }
  else if (mesh.type().cell_type() == CellType::tetrahedron)
  {
    LoadVector3DLinearForm b(f);
    assemble(x, b, mesh);
  }
  else
  {
    error("Unknown mesh type.");
  } 
*/
}
//-----------------------------------------------------------------------------
