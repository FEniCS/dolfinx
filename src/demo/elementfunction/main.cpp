// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Thanks to David Heintz for the reference matrices.

#include <iostream>
#include <dolfin.h>


using namespace dolfin;

namespace dolfin {
  
class DummyPDE : public PDE {
public:
  DummyPDE(Function::Vector& wprevious) : PDE(2, 1), wp(1)
  {
    add(wp, wprevious);
  }

  real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
  {
    //ElementFunction w = u(0).ddx();
    ElementFunction wx = wp(0).ddx();
    ElementFunction wy = wp(0).ddy();
    ElementFunction wz = wp(0).ddz();

    std::cout << "cell(): " << cell_->id() << std::endl;
    std::cout << "ddx(0): " << wx(cell_->coord(0)) << std::endl;
    std::cout << "ddy(0): " << wy(cell_->coord(0)) << std::endl;
    std::cout << "ddz(0): " << wz(cell_->coord(0)) << std::endl;

    return 0;
  }

  real rhs(ShapeFunction::Vector& v)
  {
    return 0;
  }
  
protected:
  ElementFunction::Vector wp;
};

}

int main()
{
  dolfin_set("output", "plain text");

  Matrix A;
  Vector b;

  // Load reference mesh and matrices
  //Mesh mesh("trimesh-1.xml.gz");
  Mesh mesh("mytrimesh.xml.gz");

  Vector x0;

  Function::Vector u0(mesh, x0, 1);
  x0 = 0.0;
  x0(0) = 1.0;

  DummyPDE pde(u0);

  FEM::assemble(pde, mesh, A, b);

  return 0;
}
