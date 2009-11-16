// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-21
// Last changed: 2009-10-05

#include <dolfin/fem/assemble.h>
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
void MatrixFactory::compute_mass_matrix(GenericMatrix& A, Mesh& mesh)
{
  warning("Using default dof map in MatrixFactory");
  if (mesh.type().cell_type() == CellType::triangle)
  {
    MassMatrix2D::FunctionSpace V(mesh);
    MassMatrix2D::BilinearForm a(V, V);
    assemble(A, a);
  }
  else if (mesh.type().cell_type() == CellType::tetrahedron)
  {
    MassMatrix3D::FunctionSpace V(mesh);
    MassMatrix3D::BilinearForm a(V, V);
    assemble(A, a);
  }
  else
    error("Unknown mesh type.");
}
//-----------------------------------------------------------------------------
void MatrixFactory::compute_stiffness_matrix(GenericMatrix& A,
                                             Mesh& mesh,
                                             double c)
{
  // Create constant
  Constant f(c);

  warning("Using default dof map in MatrixFactory");
  if (mesh.type().cell_type() == CellType::triangle)
  {
    StiffnessMatrix2D::FunctionSpace V(mesh);
    StiffnessMatrix2D::BilinearForm a(V, V);
    a.c = f;
    assemble(A, a);
  }
  else if (mesh.type().cell_type() == CellType::tetrahedron)
  {
    StiffnessMatrix3D::FunctionSpace V(mesh);
    StiffnessMatrix3D::BilinearForm a(V, V);
    a.c = f;
    assemble(A, a);
  }
  else
    error("Unknown mesh type.");

}
//-----------------------------------------------------------------------------
void MatrixFactory::compute_convection_matrix(GenericMatrix& A,
                                              Mesh& mesh,
                                              double cx, double cy, double cz)
{
  error("MatrixFactory need to be updated for new Function interface.");

  Constant fx(cx);
  Constant fy(cy);
  Constant fz(cz);

  warning("Using default dof map in MatrixFactory");
  if (mesh.type().cell_type() == CellType::triangle)
  {
    ConvectionMatrix2D::FunctionSpace V(mesh);
    ConvectionMatrix2D::BilinearForm a(V, V);
    a.cx = fx;
    a.cy = fy;
    assemble(A, a);
  }
  else if (mesh.type().cell_type() == CellType::tetrahedron)
  {
    ConvectionMatrix3D::FunctionSpace V(mesh);
    ConvectionMatrix3D::BilinearForm a(V, V);
    a.cx = fx;
    a.cy = fy;
    a.cz = fz;
    assemble(A, a);
  }
  else
  {
    error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
void MatrixFactory::compute_load_vector(GenericVector& x, Mesh& mesh, double c)
{
  error("MatrixFactory need to be updated for new Function interface.");

  Constant f(c);

  error("MF forms need to be updated to new mesh format.");
  warning("Using default dof map in MatrixFactory");
  if (mesh.type().cell_type() == CellType::triangle)
  {
    LoadVector2D::FunctionSpace V(mesh);
    LoadVector2D::LinearForm b(V);
    b.c = f;
    assemble(x, b);
  }
  else if (mesh.type().cell_type() == CellType::tetrahedron)
  {
    LoadVector2D::FunctionSpace V(mesh);
    LoadVector3D::LinearForm b(V);
    b.c = f;
    assemble(x, b);
  }
  else
  {
    error("Unknown mesh type.");
  }
}
//-----------------------------------------------------------------------------
