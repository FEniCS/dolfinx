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

// #include "ffc-forms/MassMatrix2D.h"
// #include "ffc-forms/MassMatrix3D.h"
// #include "ffc-forms/StiffnessMatrix2D.h"
// #include "ffc-forms/StiffnessMatrix3D.h"
// #include "ffc-forms/ConvectionMatrix2D.h"
// #include "ffc-forms/ConvectionMatrix3D.h"
// #include "ffc-forms/LoadVector2D.h"
// #include "ffc-forms/LoadVector3D.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MatrixFactory::computeMassMatrix(GenericMatrix& A, Mesh& mesh)
{
  dolfin_error("MF forms need to be updated to new mesh format.");
//   if ( mesh.type().cellType() == CellType::triangle )
//   {
//     MassMatrix2D::BilinearForm a;
//     FEM::assemble(a, A, mesh);
//   }
//   else if ( mesh.type().cellType() == CellType::tetrahedron )
//   {
//     MassMatrix3D::BilinearForm a;
//     FEM::assemble(a, A, mesh);
//   }
//   else
//   {
//     dolfin_error("Unknown mesh type.");
//   }
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeStiffnessMatrix(GenericMatrix& A, Mesh& mesh,
					   real c)
{
  dolfin_error("MF forms need to be updated to new mesh format.");
//   if ( mesh.type().cellType() == CellType::triangle )
//   {
//     StiffnessMatrix2D::BilinearForm a(c);
//     FEM::assemble(a, A, mesh);
//   }
//   else if ( mesh.type().cellType() == CellType::tetrahedron )
//   {
//     StiffnessMatrix3D::BilinearForm a(c);
//     FEM::assemble(a, A, mesh);
//   }
//   else
//   {
//     dolfin_error("Unknown mesh type.");
//   } 
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeConvectionMatrix(GenericMatrix& A, Mesh& mesh,
					    real cx, real cy, real cz)
{
  dolfin_error("MF forms need to be updated to new mesh format.");
//   if ( mesh.type().cellType() == CellType::triangle )
//   {
//     ConvectionMatrix2D::BilinearForm a(cx, cy);
//     FEM::assemble(a, A, mesh);
//   }
//   else if ( mesh.type().cellType() == CellType::tetrahedron )
//   {
//     ConvectionMatrix3D::BilinearForm a(cx, cy, cz);
//     FEM::assemble(a, A, mesh);
//   }
//   else
//   {
//     dolfin_error("Unknown mesh type.");
//   }
}
//-----------------------------------------------------------------------------
void MatrixFactory::computeLoadVector(GenericVector& x, Mesh& mesh, real c)
{
  dolfin_error("MF forms need to be updated to new mesh format.");
//   if ( mesh.type().cellType() == CellType::triangle )
//   {
//     LoadVector2D::LinearForm b(c);
//     FEM::assemble(b, x, mesh);
//   }
//   else if ( mesh.type().cellType() == CellType::tetrahedron )
//   {
//     LoadVector3D::LinearForm b(c);
//     FEM::assemble(b, x, mesh);
//   }
//   else
//   {
//     dolfin_error("Unknown mesh type.");
//   } 
}
//-----------------------------------------------------------------------------
