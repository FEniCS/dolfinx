// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2006-06-16
//
// This file is used for testing out new features implemented in the
// library, which means that the contents of this file is constantly
// changing. Anything can be thrown into this file at any time. Use
// this for simple tests that are not suitable to be implemented as
// demos in src/demo.

#include <dolfin.h>
#include <dolfin/Poisson2D.h>
#include "L2Norm.h"

using namespace dolfin;

void testPreconditioner()
{
  dolfin_info("--- Testing preconditioner ---");

/*
  class MyPreconditioner : public Preconditioner
  {
  public:
    
    MyPreconditioner(const Vector& m) : m(m) {}
    
    void solve(Vector& x, const Vector& b)
    {
      dolfin_info("Calling preconditioner");
      for (unsigned int i = 0; i < x.size(); i++)
      {
        x(i) = b(i) / m(i);
      }
    }
    
  private:
    
    const Vector& m;
    
  };
  
  UnitSquare mesh(2, 2);
  MassMatrix M(mesh);
  Vector x;
  Vector b(M.size(0));
  b = 1.0;
  Vector m;
  FEM::lump(M, m);


  MyPreconditioner pc(m); 
  GMRES solver(pc);
  solver.solve(M, x, b);
  
  x.disp();
*/
}

void testOutputVector()
{
  dolfin_info("--- Testing output of vector ---");

  Vector x(10);
  for (unsigned int i = 0; i < 10; i++)
    x(i) = 1.0 / static_cast<real>(i + 1);

  File mfile("vector.m");
  mfile << x;

  File xmlfile("vector.xml");
  xmlfile << x;
}

void testOutputMatrix()
{
  dolfin_info("--- Testing output of matrix ---");

  UnitSquare mesh(2, 2);
  MassMatrix M(mesh);

  File mfile("matrix.m");
  mfile << M;

  File xmlfile("matrix.xml");
  xmlfile << M;
}

void testOutputMesh()
{
  dolfin_info("--- Testing output of mesh ---");

  UnitCube mesh(8, 8, 8);
  File file("mesh.m");
  file << mesh;
}

void testOutputFiniteElementSpec()
{
  dolfin_info("--- Testing output of finite element specification ---");

  FiniteElementSpec spec("Lagrange", "triangle", 1, 1);
  File file("finiteelement.xml");
  file << spec;
}

void testOutputFunction()
{
  dolfin_info("--- Testing output of function ---");

  UnitSquare mesh(4, 4);
  Vector x(mesh.numVertices());
  P1tri element;

  Function f(x, mesh, element);

  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    Point & p = v->coord();
    x(v->id()) = sin(2.0*p.x) * cos(3.0*p.y);
  }

  File file("function.xml");
  file << f;
}

void testOutputMultiple()
{
  dolfin_info("--- Testing input of multiple XML objects ---");

  Vector x(3);
  Matrix A(3, 3);
  A(1, 1) = 1.0;

  File file("multiple.xml");
  file << x;
  file << A;
  file << A;
}

void testInputFunction()
{
  dolfin_info("--- Testing input of function ---");

  Function f;
  File file("function.xml");
  file >> f;

  for (VertexIterator v(f.mesh()); !v.end(); ++v)
    cout << v->coord() << ": f = " << f(*v) << endl;
}

void testFunctional()
{
  dolfin_info("--- Testing functional ---");

  class MyFunction0 : public Function
  {
  public:
    
    real eval(const Point& p, unsigned int i)
    {
      return p.x;
    }
    
  };
  
  class MyFunction1 : public Function
  {
  public:
    
    real eval(const Point& p, unsigned int i)
    {
      return p.y;
    }
    
  };

  dolfin_info("Computing L2 norm of f(x, y) = x - y on the unit square.");

  UnitSquare mesh(16, 16);
  MyFunction0 f;
  MyFunction1 g;
  L2Norm::LinearForm L(f, g);
  Vector b;
  FEM::assemble(L, b, mesh);
  real norm = sqrt(b.sum());

  dolfin_info("Result:   %.15f", norm);
  dolfin_info("Analytic: 0.408248290463863");
}

void testRandom()
{
  dolfin_info("--- Testing random ---");

  cout << "Random numbers: ";
  cout << dolfin::rand() << " ";
  cout << dolfin::rand() << " ";
  cout << dolfin::rand() << endl;
}

void testProgress()
{
  dolfin_info("--- Testing progress ---");

  Progress p("Testing progress bar", 500);
  for (unsigned int i = 0; i < 500; i++)
    p++;
}

void testParameters()
{
  dolfin_info("--- Testing parameters ---");

  class Foo : public Parametrized {};
  class Bar : public Parametrized {};

  Foo foo;
  Bar bar;

  bar.set("parent", foo);

  foo.set("tolerance", 0.00007);
  foo.set("solution file name", "foo.tmp");

  bar.set("solution file name", "bar.tmp");
  bar.set("my parameter", "bar");

  cout << "global parameters:" << endl;
  cout << "  " << get("method")    << endl;
  cout << "  " << get("tolerance") << endl;
  cout << "  " << get("solution file name") << endl;

  cout << "parameters for foo: "       << endl;
  cout << "  " << foo.get("method")    << endl;
  cout << "  " << foo.get("tolerance") << endl;
  cout << "  " << foo.get("solution file name") << endl;

  cout << "parameters for bar: "          << endl;
  cout << "  " << bar.get("method")       << endl;
  cout << "  " << bar.get("tolerance")    << endl;
  cout << "  " << bar.get("solution file name")    << endl;
  cout << "  " << bar.get("my parameter") << endl;
}

void testMakeElement()
{
  dolfin_info("--- Testing creation of element from spec ---");
  
  FiniteElement* P1 = FiniteElement::makeElement("Lagrange", "triangle", 1);
  FiniteElement* P2 = FiniteElement::makeElement("Lagrange", "triangle", 2);

  if ( P1 ) delete P1;
  if ( P2 ) delete P2;
}

void testMeshRefinement()
{
  dolfin_info("--- Testing mesh refinement ---");
  
  UnitSquare mesh(2, 2);

  File file0("mesh0.pvd");
  File file1("mesh1.pvd");
  File file2("mesh2.pvd");

  //file0 << mesh;

  // Mark one cell for refinement
  //mesh.cell(0).mark();
  //mesh.refine();
  //file1 << mesh;

  // Mark another cell in parent for refinement
  //Mesh mesh1(mesh);
  //parent.cell(7).mark();
  // parent.refine();
  //file2 << mesh;
}

void testuBlasSparseMatrix()
{
  dolfin_info("--- Testing sparse matrices ---");
  
  // Boundary condition
  class MyBC : public BoundaryCondition
  {
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      if ( std::abs(p.x - 1.0) < DOLFIN_EPS )
        value = 0.0;
    }
  };

   class Source : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return p.x*sin(p.y);
    }
  };

  UnitSquare mesh(10,10);

  Source f;
  MyBC bc;
  Poisson2D::BilinearForm a;
  Poisson2D::LinearForm L(f);

  uBlasSparseMatrix A;
  DenseVector b;
  DenseVector x;

#ifdef HAVE_PETSC_H

  PETScSparseMatrix As;
  PETScVector bs;
  PETScVector xs;

  // Assemble and solve using PETSc
  dolfin_log(false);
  FEM::assemble(a, L, As, bs, mesh, bc);
  dolfin_log(true);

//  PETScKrylovSolver solver(PETScKrylovSolver::bicgstab);
  PETScKrylovSolver solver(PETScKrylovSolver::gmres);
  tic();
  solver.solve(As, xs, bs);
  real t_petsc = toc();  
  cout << "norm(x) " << xs.norm() << endl;  
  cout << "PETSc Krylov solve time = " << t_petsc << endl;  

#endif

  dolfin_log(false);
  tic();
  FEM::assemble(a, L, A, b, mesh, bc);
  dolfin_log(true);
  cout << "uBlas assembly time = " << toc() << endl;  

  //uBlasKrylovSolver ublas_solver(uBlasKrylovSolver::bicgstab);
  uBlasKrylovSolver ublas_solver(uBlasKrylovSolver::gmres);
  uBlasILUPreconditioner pc(A);

  x = b;
  x = 0.0;  
  tic();
//  ublas_solver.solve(A, x, b, pc);
  real t_ublas = toc();  
  cout << "norm(x) " << x.norm() << endl;  
  cout << "uBlas Krylov solve time = " << t_ublas << endl;  

#ifdef HAVE_PETSC_H
  cout << "Factor ( < 1 mean uBlas is faster ) " << t_ublas/t_petsc << endl;  
#endif

}

void benchOldMesh()
{
  int num_reads = 1000;
  int num_refinements = 8;
  int num_iterations = 100;

  dolfin_log(false);

  tic();
  for (int i = 0; i < num_reads; i++)
  {
    Mesh mesh("oldmesh.xml.gz");
  }
  real t0 = toc();

  Mesh mesh("oldmesh.xml.gz");
  tic();
  for (int i = 0; i < num_refinements; i++)
    mesh.refineUniformly();
  real t1 = toc();

  tic();
  int sum = 0;
  for (int i = 0; i < num_iterations; i++)
  {

    for (CellIterator c(mesh); !c.end(); ++c)
      for (VertexIterator v(c); !v.end(); ++v)
	sum += v->id();
  }
  cout << "sum = " << sum << endl;
  real t2 = toc();

  dolfin_log(true);

  cout << mesh << endl;

  dolfin_info("Reading and initializing mesh %d times:    %.3g s", num_reads, t0);
  dolfin_info("Refining mesh uniformly %d times:            %.3g s", num_refinements, t1);
  dolfin_info("Iterating over mesh connectivity %d times: %.3g s", num_iterations, t2);

  cout << "Sleeping so you have time to measure the memory usage of this process..." << endl;
  delay(5.0);
}

void benchNewMesh()
{
  int num_reads = 1000;
  int num_refinements = 12;
  int num_iterations = 100;

  dolfin_log(false);

  tic();
  for (int i = 0; i < num_reads; i++)
  {
    NewMesh mesh("newmesh.xml.gz");
  }
  real t0 = toc();

  NewMesh mesh("newmesh.xml.gz");
  tic();
  for (int i = 0; i < num_refinements; i++)
    mesh.refine();
  real t1 = toc();

  tic();
  int sum = 0;
  for (int i = 0; i < num_iterations; i++)
  {

    for (NewCellIterator c(mesh); !c.end(); ++c)
      for (NewVertexIterator v(c); !v.end(); ++v)
	sum += v->index();
  }
  cout << "sum = " << sum << endl;
  real t2 = toc();

  dolfin_log(true);

  cout << mesh << endl;

  dolfin_info("Reading and initializing mesh %d times:    %.3g s", num_reads, t0);
  dolfin_info("Refining mesh uniformly %d times:            %.3g s", num_refinements, t1);
  dolfin_info("Iterating over mesh connectivity %d times: %.3g s", num_iterations, t2);

  cout << "Sleeping so you have time to measure the memory usage of this process..." << endl;
  delay(5.0);
}

void testNewMesh()
{
  NewMesh mesh("newmesh.xml.gz");
  cout << mesh << endl;

  // Compute all entities and connectivity, otherwise
  // handled automatically by the mesh iterators.
  mesh.init();
  mesh.disp();

  // Iterate over all entities of dimension 0 (vertices)
  for (MeshEntityIterator e(mesh, 0); !e.end(); ++e)
  {
    cout << e << endl;
    cout << *e << endl;
  }

  cout << endl;

  // Same thing with named iterators
  for (NewVertexIterator v(mesh); !v.end(); ++v)
  {
    cout << v << endl;
    cout << *v << endl;
  }

  cout << endl;

  // Nested iteration
  for (MeshEntityIterator e2(mesh, 2); !e2.end(); ++e2)
  {
    cout << *e2 << endl;
    for (MeshEntityIterator e0(e2, 0); !e0.end(); ++e0)
    {
      cout << "  " << *e0 << endl;
      for (MeshEntityIterator e1(e0, 1); !e1.end(); ++ e1)
	cout << "    " << *e1 << endl;
    }
  }

  cout << endl;
  
  // Same thing with named iterators
  for (NewCellIterator c(mesh); !c.end(); ++c)
  {
    cout << *c << endl;
    for (NewVertexIterator v(c); !v.end(); ++v)
    {
      cout << "  " << *v << endl;
      for (NewEdgeIterator e(v); !e.end(); ++ e)
	cout << "    " << *e << endl;
    }
  }

  // Test uniform mesh refinement
  mesh.refine();
  mesh.refine();
  
  // Test computation of boundary
  BoundaryMesh boundary(mesh);
}

void testVectorAccess()
{
  // Check speed of accessing individual elements for
  // different types of vectors

  unsigned int N = 10000;
  unsigned int num_repetitions = 0;
  real t = 0.0;
  real tmp = 0.0;

  PETScVector x1(N);      // PETSc vector
  DenseVector x2(N);      // uBlas vector
  Array<real> x3(N);      // STL vector
  real* x4 = new real[N]; // Plain C array

  // PETSc

  num_repetitions = 50;
  tic();
  for (unsigned int n = 0; n < num_repetitions; n++)
  {
    for (unsigned int i = 0; i < N; i++)
      x1(i) = 1.0;
  }
  t = toc();
  dolfin_info("Setting values in PETSc vector:   %.3g", t / static_cast<real>(num_repetitions));

  num_repetitions = 1000;
  tic();
  for (unsigned int n = 0; n < num_repetitions; n++)
  {
    for (unsigned int i = 0; i < N; i++)
      tmp = x1(i);
  }
  t = toc();
  dolfin_info("Accessing values in PETSc vector: %.3g", t / static_cast<real>(num_repetitions));

  // uBlas

  num_repetitions = 100000;
  tic();
  for (unsigned int n = 0; n < num_repetitions; n++)
  {
    for (unsigned int i = 0; i < N; i++)
      x2(i) = 1.0;
  }
  t = toc();
  dolfin_info("Setting values in uBlas vector:   %.3g", t / static_cast<real>(num_repetitions));

  num_repetitions = 100000;
  tic();
  for (unsigned int n = 0; n < num_repetitions; n++)
  {
    for (unsigned int i = 0; i < N; i++)
      tmp = x2(i);
  }
  t = toc();
  dolfin_info("Accessing values in uBlas vector: %.3g", t / static_cast<real>(num_repetitions));

  // STL

  num_repetitions = 100000;
  tic();
  for (unsigned int n = 0; n < num_repetitions; n++)
  {
    for (unsigned int i = 0; i < N; i++)
      x3[i] = 1.0;
  }
  t = toc();
  dolfin_info("Setting values in STL vector:     %.3g", t / static_cast<real>(num_repetitions));

  num_repetitions = 100000;
  tic();
  for (unsigned int n = 0; n < num_repetitions; n++)
  {
    for (unsigned int i = 0; i < N; i++)
      tmp = x3[i];
  }
  t = toc();
  dolfin_info("Accessing values in STL vector:   %.3g", t / static_cast<real>(num_repetitions));

  // Plain C array

  num_repetitions = 100000;
  tic();
  for (unsigned int n = 0; n < num_repetitions; n++)
  {
    for (unsigned int i = 0; i < N; i++)
      x4[i] = 1.0;
  }
  t = toc();
  dolfin_info("Setting values in C array:        %.3g", t / static_cast<real>(num_repetitions));

  num_repetitions = 100000;
  tic();
  for (unsigned int n = 0; n < num_repetitions; n++)
  {
    for (unsigned int i = 0; i < N; i++)
      tmp = x4[i];
  }
  t = toc();
  dolfin_info("Accessing values in C array:      %.3g", t / static_cast<real>(num_repetitions));
}

/*
void testSepcialization()
{

template<class T>
class Testing
  {
    public:
    
  }
}
*/


int main(int argc, char* argv[])
{
/*
  testPreconditioner();
  testOutputVector();
  testOutputMatrix();
  testOutputMesh();
  testOutputFiniteElementSpec();
  testOutputFunction();
  testOutputMultiple();
  testInputFunction();
  testFunctional();
  testRandom();
  testProgress();
  testParameters();
  testMakeElement();
  testMeshRefinement();
  testDenseMatrix();
  testDenseLUsolve();
  testGenericMatrix();
  testGenericMatrix();
*/
  testuBlasSparseMatrix();
//  testOutputMatrix();

  //testNewMesh();
  
  //benchOldMesh();
  //benchNewMesh();
  //testVectorAccess();

  return 0;
}
