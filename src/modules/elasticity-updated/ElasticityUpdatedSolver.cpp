// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004, 2005.

//#include <iostream>
#include <sstream>
#include <iomanip>

#include "dolfin/timeinfo.h"
#include "dolfin/ElasticityUpdatedSolver.h"
#include "dolfin/ElasticityUpdated.h"
#include "dolfin/ElasticityUpdatedSigma0.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ElasticityUpdatedSolver::ElasticityUpdatedSolver(Mesh& mesh, 
				   Function& f,
				   Function& u0, Function& v0,
				   real E, real nu,
				   BoundaryCondition& bc,
				   real k, real T)
  : mesh(mesh), f(f), u0(u0), v0(v0), E(E), nu(nu), bc(bc), k(k), T(T),
    counter(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::solve()
{
  real t = 0.0;  // current time
//   real T = 5.0;  // final time
//   real k = 0.01; // time step

//   real E = 10.0;
//   real nu = 0.3;
  
  real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  real mu = E / (2 * (1 + nu));
  
//   real lambda = 1.0;
//   real mu = 1.0;

  // Create element
  ElasticityUpdated::LinearForm::TestElement element1;
  ElasticityUpdatedSigma0::LinearForm::TestElement element2;

  Matrix A, M, A2;
  Vector x10, x11, x20, x21, x11old, x21old, b, m, xtmp1, xtmp2, stepresidual;
  Vector xsigma00, xsigma01, xsigma10, xsigma11, xsigma20, xsigma21;

  Function v1(x21, mesh, element1);
  Function sigma01(xsigma01, mesh, element2);
  Function sigma11(xsigma11, mesh, element2);
  Function sigma21(xsigma21, mesh, element2);

  // Create variational forms
  ElasticityUpdated::LinearForm Lv(sigma01, sigma11, sigma21);
  ElasticityUpdatedSigma0::LinearForm Lsigma0(v1, lambda, mu);




  File         file("elasticity.m");

  // FIXME: Temporary fix
  int Nv = 3 * mesh.noNodes();
  int Nsigma = 3 * mesh.noCells();

  x10.init(Nv);
  x11.init(Nv);
  x20.init(Nv);
  x21.init(Nv);

  x11old.init(Nv);
  x21old.init(Nv);

  xtmp1.init(Nv);
  xtmp2.init(Nv);

  xsigma00.init(Nsigma);
  xsigma01.init(Nsigma);

  stepresidual.init(Nv);

  // Assemble v vector
  FEM::assemble(Lv, x21, mesh);

  // Assemble sigma0 vector
  FEM::assemble(Lsigma0, xsigma01, mesh);

  // Save the solution
  save(mesh, file);

  // Start a progress session
  Progress p("Time-stepping");
  
  // Start time-stepping
  while ( t < T ) {
  
    cout << "x11: " << endl;
    x11.disp();
    
    cout << "x21: " << endl;
    x21.disp();

    cout << "xsigma01: " << endl;
    xsigma01.disp();

    // Make time step
    x10 = x11;
    x20 = x21;
    xsigma00 = xsigma01;

    // Assemble v vector
    FEM::assemble(Lv, x21, mesh);
    
    // Assemble sigma0 vector
    FEM::assemble(Lsigma0, xsigma01, mesh);

    // Save the solution
    save(mesh, file);

    counter++;

    t += k;
    f.set(t);

    // Update progress
    p = t / T;

  }
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::save(Mesh& mesh, File& solutionfile)
{
  if(counter % (int)(1.0 / 33.0 / k) == 0)
  {
    std::ostringstream fileid, filename;
    fileid.fill('0');
    fileid.width(6);
    
    fileid << counter;
    
    filename << "mesh" << fileid.str() << ".xml.gz";
    
    cout << "writing: " << filename.str() << endl;
    
    std::string foo = filename.str();
    const char *fname = foo.c_str();
    
    File meshfile(fname);
    
    meshfile << mesh;
    
  }
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::solve(Mesh& mesh,
			     Function& f,
			     Function& u0, Function& v0,
			     real E, real nu,
			     BoundaryCondition& bc,
			     real k, real T)
{
  ElasticityUpdatedSolver solver(mesh, f, u0, v0, E, nu, bc, k, T);
  solver.solve();
}
//-----------------------------------------------------------------------------
