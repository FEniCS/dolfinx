// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <sstream>
#include <iomanip>


#include "ElasticityUpdatedSolver.h"
#include "ElasticityUpdated.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ElasticityUpdatedSolver::ElasticityUpdatedSolver(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::REAL,      "final time",  1.0);
  dolfin_parameter(Parameter::REAL,      "time step",   0.1);
  dolfin_parameter(Parameter::VFUNCTION, "source",      0);
  dolfin_parameter(Parameter::VFUNCTION, "initial velocity",      0);
}
//-----------------------------------------------------------------------------
const char* ElasticityUpdatedSolver::description()
{
  return "elasticity-updated";
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::solve()
{
  Matrix A;
  Vector residual, stepresidual;
  Vector x10, x11, x20, x21, xtot, xzero, b, m, xcomp, tmp;
  
  std::cerr << "Elasticity updated" << std::endl;

  Function::Vector u0(mesh, x10, 3);
  Function::Vector u1(mesh, x11, 3);
  Function::Vector w0(mesh, x20, 3);
  Function::Vector w1(mesh, x21, 3);

  Function::Vector f("source", 3);
  Function::Vector v0("initial velocity", 3);
  
  ElasticityUpdated   elasticity(f, w0, w1);
  KrylovSolver solver;
  File solutionfile("ElasticityUpdated.m");
  
  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  // Initialize matrices

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    Matrix *sigma0i, *sigma1i, *F0i, *F1i, *vsigmai;

    sigma0i = new Matrix(3, 3, Matrix::dense);
    sigma1i = new Matrix(3, 3, Matrix::dense);
    F0i = new Matrix(3, 3, Matrix::dense);
    F1i = new Matrix(3, 3, Matrix::dense);
    vsigmai = new Matrix(3, 3, Matrix::dense);

    F0i->ident(0);
    F0i->ident(1);
    F0i->ident(2);

    F1i->ident(0);
    F1i->ident(1);
    F1i->ident(2);

    elasticity.sigma0array.push_back(sigma0i);
    elasticity.sigma1array.push_back(sigma1i);
    elasticity.F0array.push_back(F0i);
    elasticity.F1array.push_back(F1i);
    elasticity.vsigmaarray.push_back(vsigmai);
  }

  elasticity.computedsigma.resize(mesh.noCells());

  // Set initial velocities
  for (NodeIterator n(&mesh); !n.end(); ++n)
  {
    int id = (*n).id();
    
    real v0x, v0y, v0z;

    v0x = v0(0)(n->coord().x, n->coord().y, n->coord().z, 0.0);
    v0y = v0(1)(n->coord().x, n->coord().y, n->coord().z, 0.0);
    v0z = v0(2)(n->coord().x, n->coord().y, n->coord().z, 0.0);

    x21(id * 3 + 0) = v0x; 
    x21(id * 3 + 1) = v0y; 
    x21(id * 3 + 2) = v0z; 
  }
  
  m = x21;

  elasticity.k = k;
  FEM::assemble(elasticity, mesh, A);

  cout << "Mass Matrix:" << endl;

  //A.show();

  A.lump(m);

  // Start a progress session
  Progress p("Time-stepping");
  
  int counter = 0;

  // Start time-stepping
  while ( t < T )
  {
    if(counter % 33 == 0)
    {
      Function::Vector uzero(mesh, xzero, 3);
      
      std::ostringstream fileid, filename;
      fileid.fill('0');
      fileid.width(6);
      
      fileid << counter;
      
      //filename << "mesh" << fileid.str() << ".m";
      filename << "mesh" << fileid.str() << ".xml.gz";
      
      std::cout << "writing: " << filename.str() << std::endl;
      
      std::string foo = filename.str();
      const char *fname = foo.c_str();
      
      File meshfile(fname);
      
      //meshfile << uzero;
      meshfile << mesh;
    }

    if(counter % 33 == 0)
    {
      solutionfile << u1;
    }

      
    counter++;

    // Make time step
    t += k;

    /*
    cout << "before: " << endl;

    cout << "x10: " << endl;
    x10.show();

    cout << "x11: " << endl;
    x11.show();

    cout << "x20: " << endl;
    x20.show();

    cout << "x21: " << endl;
    x21.show();
    */

    x10 = x11;
    x20 = x21;

    elasticity.k = k;
    elasticity.t = t;

    stepresidual.init(x21.size());

    for(int stepiters = 0; stepiters < 1; stepiters++)
    {

      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
	int id = (*cell).id();
	
	elasticity.computedsigma[id] = false;
      }

      // Assemble
      //FEM::assemble(elasticity, mesh, A);
      FEM::assemble(elasticity, mesh, b);
      
      //cout << "A:" << endl;
      
      //A.show();
      
      //cout << "b:" << endl;
      //b.show();
      
      // Lump and solve
      
      ///*
      //A.lump(m);
      
      for(int i = 0; i < m.size(); i++)
      {
	//x21(i) = b(i) / m(i);
	stepresidual(i) = -x21(i) + x20(i) + b(i) / m(i);
      }
      //*/
      
      // Solve the linear system
      /*
      //x21 = 0;
      
      //solver.solve(A, x21, b);
      stepresidual = 0;
      
      solver.solve(A, stepresidual, b);
      */

      cout << "step residual: " << stepresidual.norm() << endl;

      x21 += stepresidual;
      
      x11 = x10;
      x11.add(k, x21);

      if(stepresidual.norm() < 1e-5)
      {
	break;
      }
    }
    
    /*
      cout << "after: " << endl;
      
      cout << "x10: " << endl;
      x10.show();
      
      cout << "x11: " << endl;
      x11.show();
      
      cout << "x20: " << endl;
      x20.show();
      
      cout << "x21: " << endl;
      x21.show();
    */
    
    
    //Update the mesh

    ///*

    for (NodeIterator n(&mesh); !n.end(); ++n)
    {
      int id = (*n).id();

      //std::cout << "node id: " << id << std::endl;
      (*n).coord().x += x11(3 * id + 0) - x10(3 * id + 0);
      (*n).coord().y += x11(3 * id + 1) - x10(3 * id + 1);
      (*n).coord().z += x11(3 * id + 2) - x10(3 * id + 2);

      //x11(3 * id + 0) = 0;
      //x11(3 * id + 1) = 0;
      //x11(3 * id + 2) = 0;
    }

    //*/

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      Matrix *sigma0i, *sigma1i, *F0i, *F1i;
      
      sigma0i = elasticity.sigma0array[cell->id()];
      sigma1i = elasticity.sigma1array[cell->id()];

      F0i = elasticity.F0array[cell->id()];
      F1i = elasticity.F1array[cell->id()];

      *sigma0i = *sigma1i;
      *F0i = *F1i;
    }



    // Move boundary (test for smoother)

    for (NodeIterator n(&mesh); !n.end(); ++n)
    {
      int id = (*n).id();

      //if(id == 106 || id == 107 || id == 111 || id == 112 ||
      //	 id == 81  || id == 82  || id == 86  || id == 87)
      if(id == 1)
      {

	real theta = k;

	Matrix Rz(3, 3);

	Rz(0, 0) = cos(theta);
	Rz(0, 1) = -sin(theta);
	Rz(1, 0) = sin(theta);
	Rz(1, 1) = cos(theta);
	Rz(2, 2) = 1;

	/*
	Vector w(3);

	w(0) = 0;
	w(1) = 0;
	w(2) = 0.1;
	*/

	Vector center(3), r(3), p(3), v(3), rnew(3);

	p(0) = (*n).coord().x;
	p(1) = (*n).coord().y;
	p(2) = (*n).coord().z;

	center(0) = 0.375;
	center(1) = 0.375;
	center(2) = 1;

	r = p;
	r.add(-1, center);

	Rz.mult(r, rnew);

	v = rnew;
	v.add(-1, r);

	//w.cross(r, v);
	

	//std::cout << "node id: " << id << std::endl;
	//(*n).coord().x += v(0);
	//(*n).coord().y += v(1);
	//(*n).coord().z += v(2);

	//x21(id * 3 + 0) = v(0) / k; 
	//x21(id * 3 + 1) = v(1) / k; 
	//x21(id * 3 + 2) = v(2) / k; 

	//(*n).coord().x += 0.1 * k;
	//(*n).coord().y += 0.0;
	//(*n).coord().z += 0.0;

	//x21(id * 3 + 0) = 0.1; 
	//x21(id * 3 + 1) = 0.0; 
	//x21(id * 3 + 2) = 0.0; 


	
	//x11(3 * id + 0) = 0;
	//x11(3 * id + 1) = 0;
	//x11(3 * id + 2) = 0;
      }
    }



    // Update progress
    p = t / T;


  }
  //*/
}
//-----------------------------------------------------------------------------
