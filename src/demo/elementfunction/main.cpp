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
  DummyPDE(Function& wprevious) : PDE(2)
  {
    add(wp, wprevious);
  }

  real lhs(const ShapeFunction& u, const ShapeFunction& v)
  {
    return 0;
  }

  real rhs(const ShapeFunction& v)
  {
    // Derivatives
    ElementFunction wx = wp.ddx();
    ElementFunction wy = wp.ddy();
    ElementFunction wz = wp.ddz();

    std::cout << "cell(): " << cell_->id() << std::endl;
    std::cout << "wp.ddx(): " << wx(cell_->coord(0)) << std::endl;
    std::cout << "wp.ddy(): " << wy(cell_->coord(0)) << std::endl;
    std::cout << "wp.ddz(): " << wz(cell_->coord(0)) << std::endl;
    std::cout << "wp(0): " << wp(0, 0, 0, 0) << std::endl;
    std::cout << "wp(1): " << wp(1, 0, 0, 0) << std::endl;
    std::cout << "wp(2): " << wp(0, 1, 0, 0) << std::endl;

    return 0;
  }
  
protected:
  ElementFunction wp;
};

}

int main()
{
  dolfin_set("output", "plain text");

  Matrix A;
  Vector b;

  // Load reference mesh and matrices
  //Mesh mesh("trimesh-1.xml.gz");
  Mesh mesh("trimesh-1b.xml.gz");
  //Mesh mesh("triangle.xml.gz");

  Vector x0;

  Function u0(mesh, x0);
  x0 = 0.0;
  //x0(0) = 1.0;
  x0(0) = 0.01;
  x0(1) = 0.02;
  x0(2) = 0.03;
  x0(3) = 0.04;

  DummyPDE pde(u0);

  FEM::assemble(pde, mesh, A, b);

  Matrix A1(3, 3), A2(3, 3);
  Vector c1(3), c2(3), v1(3), v2(3);

  // Assume linear basis functions
  // Assume 4 nodes in mesh (2 triangles)


  for(int i = 0; i < 3; i++)
  {
    A1(i, 0) = 1.0;
    A1(i, 1) = mesh.cell(0).node(i).coord().x;
    A1(i, 2) = mesh.cell(0).node(i).coord().y;

    v1(i) = x0(mesh.cell(0).node(i).id());

    A2(i, 0) = 1.0;
    A2(i, 1) = mesh.cell(1).node(i).coord().x;
    A2(i, 2) = mesh.cell(1).node(i).coord().y;

    v2(i) = x0(mesh.cell(1).node(i).id());
  }

  KrylovSolver solver;

  solver.solve(A1, c1, v1);
  solver.solve(A2, c2, v2);

  cout << "Comparing with manually computed derivatives:" << endl;

  cout << "ddx cell 0: " << c1(1) << endl;
  cout << "ddy cell 0: " << c1(2) << endl;
  cout << "ddx cell 1: " << c2(1) << endl;
  cout << "ddy cell 1: " << c2(2) << endl;


  return 0;
}
