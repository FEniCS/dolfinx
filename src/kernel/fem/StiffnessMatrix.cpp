// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/PDE.h>
#include <dolfin/FEM.h>
#include <dolfin/StiffnessMatrix.h>

using namespace dolfin;

namespace dolfin
{

  // The variational form for the stiffness matrix
  class StiffnessForm : public PDE
  {
  public:
    
    StiffnessForm() : PDE(3) {}

    real lhs(const ShapeFunction& u, const ShapeFunction& v)
    {
      return (grad(u),grad(v)) * dx;
    }

  };

}

//-----------------------------------------------------------------------------
StiffnessMatrix::StiffnessMatrix(Mesh& mesh, real epsilon)
  : Matrix(mesh.noNodes(), mesh.noNodes())
{
  dolfin_error("This function needs to be updated to the new format.");

  /*
  StiffnessForm form;

  FEM::assemble(form, mesh, *this);
  (*this) *= epsilon;
  */
}
//-----------------------------------------------------------------------------
