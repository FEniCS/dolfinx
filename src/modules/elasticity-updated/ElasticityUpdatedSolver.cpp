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
  Vector x10, x11, x20, x21, xtot, xzero, b, xcomp, tmp;
  
  std::cerr << "Elasticity updated2" << std::endl;

  Function::Vector u0(mesh, x10, 3);
  Function::Vector u1(mesh, x11, 3);
  Function::Vector w0(mesh, x20, 3);
  Function::Vector w1(mesh, x21, 3);

  Function::Vector f("source", 3);
  
  ElasticityUpdated   elasticity(f, w0);
  KrylovSolver solver;
  File         file("ElasticityUpdated.m");
  
  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    Matrix *sigma0i, *sigma1i;

    sigma0i = new Matrix(3, 3);
    sigma1i = new Matrix(3, 3);

    elasticity.sigma0array.push_back(sigma0i);
    elasticity.sigma1array.push_back(sigma1i);
  }

  elasticity.computedsigma.resize(mesh.noCells());

  //Matrix &sigma0 = *(elasticity.sigma0array[0]);
  //sigma0(0, 0) = 50.0;

  

  elasticity.k = k;
  FEM::assemble(elasticity, mesh, A);

  // Start a progress session
  Progress p("Time-stepping");
  
  int counter = 0;


  ///*
  x21(0 * 3 + 0) = 0.0; 
  x21(1 * 3 + 0) = 8.0; 
  x21(2 * 3 + 0) = 0.0; 
  x21(3 * 3 + 0) = 0.0; 
  //*/

  /*
  x21(0 * 3 + 0) = -1.0; 
  x21(1 * 3 + 0) = 2.0; 
  x21(2 * 3 + 0) = -1.0; 
  x21(3 * 3 + 0) = -1.0; 
  */

  // Start time-stepping
  while ( t < T ) {
    Function::Vector uzero(mesh, xzero, 3);

    std::ostringstream fileid, filename;
    fileid.fill('0');
    fileid.width(6);

    fileid << counter;

    filename << "mesh" << fileid.str() << ".m";

    std::cout << "writing: " << filename.str() << std::endl;

    std::string foo = filename.str();
    const char *fname = foo.c_str();
    
    File meshfile(fname);

    meshfile << uzero;


    counter++;

    // Make time step
    t += k;

    x10 = x11;
    //x10 = 0;
    x20 = x21;

    elasticity.k = k;
    elasticity.t = t;

    cout << "before: " << endl;

    cout << "x10: " << endl;
    x10.show();

    cout << "x11: " << endl;
    x11.show();

    cout << "x20: " << endl;
    x20.show();

    cout << "x21: " << endl;
    x21.show();


    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      int id = (*cell).id();
      
      elasticity.computedsigma[id] = false;
    }

    // Assemble
    FEM::assemble(elasticity, mesh, A);
    FEM::assemble(elasticity, mesh, b);

    //cout << "A:" << endl;

    //A.show();

    //cout << "b:" << endl;

    //b.show();

    x21 = 0;

    // Solve the linear system
    solver.solve(A, x21, b);

    x11 = x10;
    x11.add(k, x21);

    //Update the mesh

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

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      Matrix *sigma0i, *sigma1i;
      
      sigma0i = elasticity.sigma0array[cell->id()];
      sigma1i = elasticity.sigma1array[cell->id()];

      *sigma0i = *sigma1i;
    }

    // Update progress
    p = t / T;


  }
  //*/
}
//-----------------------------------------------------------------------------
