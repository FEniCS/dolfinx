// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2004-2005.
//
// First added:  2004-02-26
// Last changed: 2005

//#include <iostream>
#include <sstream>
#include <iomanip>

#include "dolfin/timeinfo.h"
#include "dolfin/ElasticitySolver.h"
#include "dolfin/Elasticity.h"
#include "dolfin/ElasticityMass.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ElasticitySolver::ElasticitySolver(Mesh& mesh, 
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
void ElasticitySolver::solve()
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

  // Create variational forms
  Elasticity::BilinearForm a(lambda, mu);
  Elasticity::LinearForm L(f);
  Elasticity::LinearForm Lu0(u0);
  Elasticity::LinearForm Lv0(v0);

  ElasticityMass::BilinearForm amass;

  // Create element
  Elasticity::BilinearForm::TrialElement element;

  Matrix A, M, A2;
  Vector x10, x11, x20, x21, x11old, x21old, b, m, xtmp1, xtmp2, stepresidual;
  
  Function u1(x11, mesh, element);
  Function w1(x21, mesh, element);

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
  FEM::assemble(a, A, mesh);

  elapsed = toc();
  dolfin_debug("Assembled matrix:");
  cout << "elapsed: " << elapsed << endl;

  dolfin_debug("Assembling vector:");
  tic();

  // Assemble load vector
  FEM::assemble(L, b, mesh);

  elapsed = toc();
  dolfin_debug("Assembled vector:");
  cout << "elapsed: " << elapsed << endl;


  //return;

  // Assemble mass matrix
  FEM::assemble(amass, M, mesh);


  // Set BC
  FEM::applyBC(A, b, mesh, element, bc);

//   cout << "A: " << endl;
//   A.disp(false);
  
  dolfin_debug("Assembled matrix:");

  // Lump mass matrix

  FEM::lump(M, m);


  // Assemble initial values
  FEM::assemble(Lu0, xtmp1, mesh);
  FEM::assemble(Lv0, xtmp2, mesh);

  for(unsigned int i = 0; i < m.size(); i++)
  {
    x11(i) = xtmp1(i) / m(i);
    x21(i) = xtmp2(i) / m(i);
  }
  


  // Save the solution
  save(mesh, u1, w1, file);

  // Start a progress session
  Progress p("Time-stepping");
  
  // Start time-stepping
  while ( t < T ) {
  
     cout << "x11: " << endl;
     x11.disp();

     cout << "x21: " << endl;
     x21.disp();


    // Make time step
    x10 = x11;
    x20 = x21;

    // Assemble load vector
    FEM::assemble(L, b, mesh);

    // Set boundary conditions
    FEM::applyBC(A, b, mesh, element, bc);

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
      //cout << "inc1: " << xtmp1.norm(Vector::linf) << endl;
      //cout << "inc2: " << xtmp2.norm(Vector::linf) << endl;
      if(std::max(xtmp1.norm(Vector::linf), xtmp2.norm(Vector::linf)) < 1e-8)
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
    save(mesh, u1, w1, file);

    counter++;

    t += k;
    f.set(t);

    // Update progress
    p = t / T;

  }
}
//-----------------------------------------------------------------------------
void ElasticitySolver::save(Mesh& mesh, Function& u, Function& v,
			    File& solutionfile)
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
    
    // Deform the mesh
    
    for (NodeIterator n(&mesh); !n.end(); ++n)
    {
      (*n).coord().x += u(*n, 0);
      (*n).coord().y += u(*n, 1);
      (*n).coord().z += u(*n, 2);
    }
    
    
    meshfile << mesh;
    
    // Undo deformation
    
    for (NodeIterator n(&mesh); !n.end(); ++n)
    {
      (*n).coord().x -= u(*n, 0);
      (*n).coord().y -= u(*n, 1);
      (*n).coord().z -= u(*n, 2);
    }
  }
}
//-----------------------------------------------------------------------------
void ElasticitySolver::solve(Mesh& mesh,
			     Function& f,
			     Function& u0, Function& v0,
			     real E, real nu,
			     BoundaryCondition& bc,
			     real k, real T)
{
  ElasticitySolver solver(mesh, f, u0, v0, E, nu, bc, k, T);
  solver.solve();
}
//-----------------------------------------------------------------------------
