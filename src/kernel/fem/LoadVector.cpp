// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/PDE.h>
#include <dolfin/FEM.h>
#include <dolfin/LoadVector.h>

using namespace dolfin;

namespace dolfin
{

  // The variational form for the load vector
  class LoadForm : public PDE
  {
  public:
    
    LoadForm() : PDE(3) {}
    
    real rhs(const ShapeFunction& v)
    {
      return 1.0*v * dx;
    }
    
  };

}

//-----------------------------------------------------------------------------
LoadVector::LoadVector(Mesh& mesh) : Vector(mesh.noNodes())
{
  LoadForm form;
  
  FEM::assemble(form, mesh, *this);
}
//-----------------------------------------------------------------------------
