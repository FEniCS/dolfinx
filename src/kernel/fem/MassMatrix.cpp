// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/PDE.h>
#include <dolfin/FEM.h>
#include <dolfin/MassMatrix.h>

using namespace dolfin;

namespace dolfin
{

  // The variational form for the mass matrix
  class MassForm : public PDE
  {
  public:
    
    MassForm() : PDE(3) {}

    real lhs(const ShapeFunction& u, const ShapeFunction& v)
    {
      return u*v * dx;
    }

  };

}

//-----------------------------------------------------------------------------
MassMatrix::MassMatrix(Mesh& mesh) : Matrix(mesh.noNodes(), mesh.noNodes())
{
  MassForm form;

  FEM::assemble(form, mesh, *this);
}
//-----------------------------------------------------------------------------
