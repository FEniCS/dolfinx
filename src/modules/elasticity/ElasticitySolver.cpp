// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#include "dolfin/timeinfo.h"
#include "dolfin/ElasticitySolver.h"
#include "dolfin/Elasticity.h"
#include "dolfin/ElasticityMass.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ElasticitySolver::ElasticitySolver(Mesh& mesh, 
				   NewFunction &f,
				   NewBoundaryCondition& bc)
  : mesh(mesh), f(f), bc(bc)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ElasticitySolver::solve()
{
  real t = 0.0;  // current time
  real T = 5.0;  // final time
  real k = 0.01; // time step

  real E = 10.0;
  real nu = 0.3;
  
  real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  real mu = E / (2 * (1 + nu));
  
//   real lambda = 1.0;
//   real mu = 1.0;

  // Create variational forms
  Elasticity::BilinearForm a(lambda, mu);
  Elasticity::LinearForm L(f);

  ElasticityMass::BilinearForm amass;

  // Create element
  Elasticity::BilinearForm::TrialElement element;

  Matrix A, M, A2;
  Vector x10, x11, x20, x21, x11old, x21old, b, m, xtmp1, xtmp2, stepresidual;
  
  NewFunction u0(x10, mesh, element);
  NewFunction u1(x11, mesh, element);
  NewFunction w0(x20, mesh, element);
  NewFunction w1(x21, mesh, element);

  File         file("elasticity.m");

  // FIXME: Temporary fix
  int N = 3 * mesh.noNodes();

  x10.init(N);
  x11.init(N);
  x20.init(N);
  x21.init(N);

  x11old.init(N);
  x21old.init(N);

  xtmp1.init(N);
  xtmp2.init(N);

  stepresidual.init(N);


  real elapsed = 0;

  dolfin_debug("Assembling matrix:");
  tic();

  // Assemble stiffness matrix
  NewFEM::assemble(a, A, mesh);

  elapsed = toc();
  dolfin_debug("Assembled matrix:");
  cout << "elapsed: " << elapsed << endl;

  dolfin_debug("Assembling vector:");
  tic();

  // Assemble load vector
  NewFEM::assemble(L, b, mesh);

  elapsed = toc();
  dolfin_debug("Assembled vector:");
  cout << "elapsed: " << elapsed << endl;


  //return;

  // Assemble mass matrix
  NewFEM::assemble(amass, M, mesh);


  // Set BC
  NewFEM::setBC(A, b, mesh, bc);

//   cout << "A: " << endl;
//   A.disp(false);
  
  dolfin_debug("Assembled matrix:");

  // Lump mass matrix

  NewFEM::lump(M, m);

//   A2 = M;

//   for(unsigned int i = 0; i < A.size(0); i++)
//   {
//     for(unsigned int j = 0; j < A.size(0); j++)
//     {
//       if(fabs(A(i, j)) > DOLFIN_EPS)
//       {
// 	A2(i, j) = A2(i, j) + k * k * A(i, j);
//       } 
//     }
//   }

//   cout << "A2: " << endl;
//   A2.disp(false);

//   cout << "M: " << endl;
//   M.disp(false);
//   cout << "m: " << endl;
//   m.disp();

  file << u1;

  // Start a progress session
  Progress p("Time-stepping");
  
  int counter = 0;

  // Start time-stepping
  while ( t < T ) {
  
    // Make time step
    x10 = x11;
    x20 = x21;

    // Assemble load vector
    NewFEM::assemble(L, b, mesh);

    // Set boundary conditions
    NewFEM::setBC(A, b, mesh, bc);


    // Fixed point iteration
    
    for(int fpiter = 0; fpiter < 50; fpiter++)
    {
      x11old = x11;
      x21old = x21;

      //Astiff.mult(x11old, xtmp1);
      A.mult(x11old, xtmp1);

      for(unsigned int i = 0; i < m.size(); i++)
      {
	stepresidual(i) = -x21(i) + x20(i) -
	  k * xtmp1(i) / m(i) + k * b(i) / m(i);
      }

      x21 += stepresidual;

      x11 = x10;
      x11.axpy(k, x21old);

      xtmp1 = x11;
      xtmp1.axpy(-1, x11old);
      xtmp2 = x21;
      xtmp2.axpy(-1, x21old);
      cout << "inc1: " << xtmp1.norm(Vector::linf) << endl;
      cout << "inc2: " << xtmp2.norm(Vector::linf) << endl;
      if(max(xtmp1.norm(Vector::linf), xtmp2.norm(Vector::linf)) < 1e-8)
      {
	cout << "fixed point iteration converged" << endl;
	break;
      }
    }

//     cout << "x11: " << endl;
//     x11.disp();

//     cout << "x21: " << endl;
//     x21.disp();

    
    // Save the solution
    if(counter % 3 == 0)
    {
      file << u1;
    }
    counter++;

    t += k;
    f.set(t);

    // Update progress
    p = t / T;

  }
}
//-----------------------------------------------------------------------------
void ElasticitySolver::solve(Mesh& mesh,
			     NewFunction& f,
			     NewBoundaryCondition& bc)
{
  ElasticitySolver solver(mesh, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
