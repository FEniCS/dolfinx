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
#include "dolfin/ElasticityUpdatedSigma1.h"
#include "dolfin/ElasticityUpdatedSigma2.h"
#include "dolfin/ElasticityUpdatedProj.h"
#include "dolfin/ElasticityUpdatedMass.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ElasticityUpdatedSolver::ElasticityUpdatedSolver(Mesh& mesh, 
				   Function& f,
				   Function& v0,
				   real E, real nu,
				   BoundaryCondition& bc,
				   real k, real T)
  : mesh(mesh), f(f), v0(v0), E(E), nu(nu), bc(bc), k(k), T(T),
    counter(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::solve()
{
  real t = 0.0;  // current time

//   real E = 10.0;
//   real nu = 0.3;
  
   real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
   real mu = E / (2 * (1 + nu));
  
//   real lambda = 1.0;
//   real mu = 2.0;

//   lambda = 0.0;

  // Create element
  ElasticityUpdated::LinearForm::TestElement element1;
  ElasticityUpdatedSigma0::LinearForm::TestElement element20;
  ElasticityUpdatedSigma1::LinearForm::TestElement element21;
  ElasticityUpdatedSigma2::LinearForm::TestElement element22;

  Matrix M;
  Vector x10, x11, x20, x21, x11old, x21old, b, m, xtmp1, xtmp2,
    xtmp01, xtmp11, xtmp21, stepresidual;
  Vector xsigma00, xsigma01, xsigma10, xsigma11, xsigma20, xsigma21;

//   Function v1(x21, mesh, element1);
//   Function sigma01(xsigma01, mesh, element2);
//   Function sigma11(xsigma11, mesh, element2);
//   Function sigma21(xsigma21, mesh, element2);

  Function v1(x21, mesh, element1);
  Function sigma01(xsigma01, mesh, element20);
  Function sigma11(xsigma11, mesh, element21);
  Function sigma21(xsigma21, mesh, element22);


  File         file("elasticity.m");

  // FIXME: Temporary fix
  int Nv = 3 * mesh.noNodes();
  int Nsigma = 3 * mesh.noCells();

  int offset = mesh.noNodes();

  x10.init(Nv);
  x11.init(Nv);
  x20.init(Nv);
  x21.init(Nv);

  x11old.init(Nv);
  x21old.init(Nv);

  xtmp1.init(Nv);
  xtmp2.init(Nv);

  xtmp01.init(Nsigma);
  xtmp11.init(Nsigma);
  xtmp21.init(Nsigma);

  xsigma00.init(Nsigma);
  xsigma01.init(Nsigma);

  xsigma10.init(Nsigma);
  xsigma11.init(Nsigma);

  xsigma20.init(Nsigma);
  xsigma21.init(Nsigma);

  xsigma01 = 0;
  xsigma11 = 0;
  xsigma21 = 0;

  stepresidual.init(Nv);

  // Set initial velocities                                                     
  for (NodeIterator n(&mesh); !n.end(); ++n)
  {
    int id = (*n).id();
    
    real v0x, v0y, v0z;
    
    v0x = v0((*n).coord(), 0);
    v0y = v0((*n).coord(), 1);
    v0z = v0((*n).coord(), 2);
    
    x21(id + 0 * offset) = v0x;
    x21(id + 1 * offset) = v0y;
    x21(id + 2 * offset) = v0z;
  }

  // Create variational forms
  //ElasticityUpdated::LinearForm Lv(sigma01, sigma11, sigma21);
  //ElasticityUpdated::LinearForm Lv;
  ElasticityUpdated::LinearForm Lv(sigma01, sigma11, sigma21);
  ElasticityUpdatedSigma0::LinearForm Lsigma0(v1, lambda, mu);
  ElasticityUpdatedSigma1::LinearForm Lsigma1(v1, lambda, mu);
  ElasticityUpdatedSigma2::LinearForm Lsigma2(v1, lambda, mu);
//   ElasticityUpdatedSigma0::LinearForm Lsigma0(v1);
//   ElasticityUpdatedSigma1::LinearForm Lsigma1(v1);
//   ElasticityUpdatedSigma2::LinearForm Lsigma2(v1);
  ElasticityUpdatedProj::LinearForm Lv0(v0);
  ElasticityUpdatedMass::BilinearForm amass;

  // Assemble mass matrix
  FEM::assemble(amass, M, mesh);

  // Lump mass matrix
  FEM::lump(M, m);

  // Assemble v vector
  //FEM::assemble(Lv, x21, mesh);

  // Assemble sigma0 vectors
//   FEM::assemble(Lsigma0, xsigma01, mesh);
//   FEM::assemble(Lsigma1, xsigma11, mesh);
//   FEM::assemble(Lsigma2, xsigma21, mesh);

  // Assemble initial velocity
  //FEM::assemble(Lv0, xtmp1, mesh);

//   for(unsigned int i = 0; i < m.size(); i++)
//   {
//     x21(i) = xtmp1(i) / m(i);
//   }

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


    // Make time step
    x10 = x11;
    x20 = x21;
    xsigma00 = xsigma01;
    xsigma10 = xsigma11;
    xsigma20 = xsigma21;

    // Assemble sigma0 vectors
    FEM::assemble(Lsigma0, xtmp01, mesh);
    FEM::assemble(Lsigma1, xtmp11, mesh);
    FEM::assemble(Lsigma2, xtmp21, mesh);

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      int id = (*cell).id();

      xtmp01(3 * id + 0) *= 1.0 / (*cell).volume();
      xtmp01(3 * id + 1) *= 1.0 / (*cell).volume();
      xtmp01(3 * id + 2) *= 1.0 / (*cell).volume();
      xtmp11(3 * id + 0) *= 1.0 / (*cell).volume();
      xtmp11(3 * id + 1) *= 1.0 / (*cell).volume();
      xtmp11(3 * id + 2) *= 1.0 / (*cell).volume();
      xtmp21(3 * id + 0) *= 1.0 / (*cell).volume();
      xtmp21(3 * id + 1) *= 1.0 / (*cell).volume();
      xtmp21(3 * id + 2) *= 1.0 / (*cell).volume();

      // Volume seems to be off by 1 / 6 for tetrahedrons
      xtmp01 *= 6.0;
      xtmp11 *= 6.0;
      xtmp21 *= 6.0;
    }



    xsigma01 = xsigma00;
    xsigma01.axpy(k, xtmp01);

    xsigma11 = xsigma10;
    xsigma11.axpy(k, xtmp11);

    xsigma21 = xsigma20;
    xsigma21.axpy(k, xtmp21);

    cout << "xsigma01: " << endl;
    xsigma01.disp();

    cout << "xsigma11: " << endl;
    xsigma11.disp();

    cout << "xsigma21: " << endl;
    xsigma21.disp();

    // Assemble v vector
    FEM::assemble(Lv, xtmp1, mesh);
    
    b = xtmp1;
    b *= k;
//     b *= 6.0;

    cout << "b: " << endl;
    b.disp();

    for(unsigned int i = 0; i < m.size(); i++)
    {
      stepresidual(i) = -x21(i) + x20(i) -
	k * xtmp1(i) / m(i) * 1.0;
    }

    x21 += stepresidual;

    x11 = x10;
    x11.axpy(k, x20);

    //Update the mesh

    for (NodeIterator n(&mesh); !n.end(); ++n)
    {
      int id = (*n).id();
      
      //std::cout << "node id: " << id << std::endl;
      (*n).coord().x += x11(id + 0 * offset) - x10(id + 0 * offset);
      (*n).coord().y += x11(id + 1 * offset) - x10(id + 1 * offset);
      (*n).coord().z += x11(id + 2 * offset) - x10(id + 2 * offset);
    }

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
			     Function& v0,
			     real E, real nu,
			     BoundaryCondition& bc,
			     real k, real T)
{
  ElasticityUpdatedSolver solver(mesh, f, v0, E, nu, bc, k, T);
  solver.solve();
}
//-----------------------------------------------------------------------------
