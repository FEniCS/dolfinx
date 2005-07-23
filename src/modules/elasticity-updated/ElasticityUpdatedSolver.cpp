// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2004-2005.
//
// First added:  2005
// Last changed: 2005

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
						 Function& rho,
						 real E, real nu, real nuv,
						 BoundaryCondition& bc,
						 real k, real T)
  : mesh(mesh), f(f), v0(v0), rho(rho), E(E), nu(nu), nuv(nuv), bc(bc), k(k),
    T(T), counter(0), lastsample(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::solve()
{
  real t = 0.0;  // current time
  
  real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  real mu = E / (2 * (1 + nu));

  real rtol = 1e+6;

  bool plasticity = false;

//   real vplast = 4e-1;
  real vplast = 0 * 1e-2;
  real yield = 1.0e0;

  cout << "lambda: " << lambda << endl;
  cout << "mu: " << mu << endl;
  
  // Create element
  ElasticityUpdated::LinearForm::TestElement element1;
  ElasticityUpdatedSigma0::LinearForm::TestElement element20;
  ElasticityUpdatedSigma1::LinearForm::TestElement element21;
  ElasticityUpdatedSigma2::LinearForm::TestElement element22;

  Matrix M;
  Vector x10, x11, x20, x21, x11old, x21old, b, m, msigma, xtmp1, xtmp2,
    xtmp01, xtmp11, xtmp21, stepresidual, stepresidual2;
  Vector xsigma00, xsigma01, xsigma10, xsigma11, xsigma20, xsigma21,
    xepsilon01, xepsilon11, xepsilon21, xsigmanorm;

  Function v1old(x20, mesh, element1);
  Function v1(x21, mesh, element1);
  Function sigma01(xsigma01, mesh, element20);
  Function sigma11(xsigma11, mesh, element21);
  Function sigma21(xsigma21, mesh, element22);
  Function sigma00(xsigma00, mesh, element20);
  Function sigma10(xsigma10, mesh, element21);
  Function sigma20(xsigma20, mesh, element22);
  Function epsilon01(xepsilon01, mesh, element20);
  Function epsilon11(xepsilon11, mesh, element21);
  Function epsilon21(xepsilon21, mesh, element22);
  Function sigmanorm(xsigmanorm, mesh, element22);


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

  msigma.init(Nsigma);

  xtmp01.init(Nsigma);
  xtmp11.init(Nsigma);
  xtmp21.init(Nsigma);

  xsigma00.init(Nsigma);
  xsigma01.init(Nsigma);

  xsigma10.init(Nsigma);
  xsigma11.init(Nsigma);

  xsigma20.init(Nsigma);
  xsigma21.init(Nsigma);

  xepsilon01.init(Nsigma);
  xepsilon11.init(Nsigma);
  xepsilon21.init(Nsigma);

  xsigmanorm.init(Nsigma);

  xsigma01 = 0;
  xsigma11 = 0;
  xsigma21 = 0;

  stepresidual.init(Nv);
  stepresidual2.init(Nv);

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

  xsigmanorm = 1.0;

  // Create variational forms
  ElasticityUpdated::LinearForm Lv(f, sigma01, sigma11, sigma21,
				   epsilon01, epsilon11, epsilon21, nuv);
  ElasticityUpdatedSigma0::LinearForm Lsigma0(v1, sigma01, sigma11, sigma21,
					      sigmanorm, lambda, mu, vplast);
  ElasticityUpdatedSigma1::LinearForm Lsigma1(v1, sigma01, sigma11, sigma21,
					      sigmanorm, lambda, mu, vplast);
  ElasticityUpdatedSigma2::LinearForm Lsigma2(v1, sigma01, sigma11, sigma21,
					      sigmanorm, lambda, mu, vplast);


  ElasticityUpdatedProj::LinearForm Lv0(v0);
  ElasticityUpdatedMass::BilinearForm amass(rho);

  // Assemble mass matrix
  FEM::assemble(amass, M, mesh);

  // Lump mass matrix
  FEM::lump(M, m);

  // Compute mass vector (sigma)
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    int id = (*cell).id();

    real factor = 1.0 / (*cell).volume(); 

    msigma(3 * id + 0) = factor;
    msigma(3 * id + 1) = factor;
    msigma(3 * id + 2) = factor;
  }

  // Save the solution
  //save(mesh, file, t);

  // Start a progress session
  Progress p("Time-stepping");
  
  // Start time-stepping
  while ( t < T ) {
  
    // Make time step
    x10 = x11;
    x20 = x21;
    xsigma00 = xsigma01;
    xsigma10 = xsigma11;
    xsigma20 = xsigma21;

    t += k;
    f.set(t);
    cout << "t: " << t << endl;
      
    for(int iter = 0; iter < 10; iter++)
    {
      
      // Compute norm of stress (sigmanorm)
      if(plasticity)
      {
	for (CellIterator cell(mesh); !cell.end(); ++cell)
	{
	  int id = (*cell).id();
	  
	  real proj = 1;
	  real norm = 0;
	  for(int i = 0; i < 3; i++)
	  {
	    norm = std::max(norm, fabs(xsigma00(3 * id + i)));
	    norm = std::max(norm, fabs(xsigma10(3 * id + i)));
	    norm = std::max(norm, fabs(xsigma20(3 * id + i)));
	  }
	  
	  if(norm > yield)
	  {
	    cout << "sigmanorm(" << id << "): " << norm << endl;
	    proj = 1.0 / norm;
	  }
	  
	  xsigmanorm(3 * id + 0) = proj;
	}
      }

      dolfin_debug("Assembling sigma vectors");
      tic();

      // Assemble sigma0 vectors
      FEM::assemble(Lsigma0, xsigma01, mesh);
      FEM::assemble(Lsigma1, xsigma11, mesh);
      FEM::assemble(Lsigma2, xsigma21, mesh);

      VecPointwiseMult(xtmp01.vec(), xsigma01.vec(), msigma.vec());
      VecPointwiseMult(xtmp11.vec(), xsigma11.vec(), msigma.vec());
      VecPointwiseMult(xtmp21.vec(), xsigma21.vec(), msigma.vec());

      xtmp01.apply();
      xtmp11.apply();
      xtmp21.apply();

      xsigma01 = xsigma00;
      xsigma01.axpy(k, xtmp01);

      xsigma11 = xsigma10;
      xsigma11.axpy(k, xtmp11);

      xsigma21 = xsigma20;
      xsigma21.axpy(k, xtmp21);

      xepsilon01 *= 1.0 / lambda;
      xepsilon11 *= 1.0 / lambda;
      xepsilon21 *= 1.0 / lambda;

      // Assemble v vector
      FEM::assemble(Lv, xtmp1, mesh);

      cout << "xtmp1: " << xtmp1.norm(Vector::linf) << endl;

      cout << "m: " << m.norm(Vector::linf) << endl;


      b = xtmp1;
      b *= k;

      VecPointwiseDivide(stepresidual.vec(), xtmp1.vec(), m.vec());
      stepresidual *= k;
      stepresidual.axpy(-1, x21);
      stepresidual.axpy(1, x20);


      x21 += stepresidual;

      x11 = x10;
      x11.axpy(k, x20);

      cout << "stepresidual: " << stepresidual.norm(Vector::linf) << endl;

      if(stepresidual.norm(Vector::linf) <= rtol && iter >= 0)
	break;
    }



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
    //save(mesh, file, t);

    // Update progress
    p = t / T;


  }
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::save(Mesh& mesh, File& solutionfile, real t)
{
  real samplefreq = 1.0 / 33.0;

  while(lastsample + samplefreq < t || t == 0.0)
  {
    std::ostringstream fileid, filename;
    fileid.fill('0');
    fileid.width(6);
    
    fileid << counter;
    
    filename << "mesh" << fileid.str() << ".xml.gz";
    
    cout << "writing: " << filename.str() << " at t: " << t << endl;
    
    std::string foo = filename.str();
    const char *fname = foo.c_str();
    
    File meshfile(fname);
    
    meshfile << mesh;
    counter++;

    cout << "lastsample: " << lastsample << " t: " << t << endl;

    lastsample = std::min(t, lastsample + samplefreq);

    if(t == 0.0)
    {
      break;
    }
  }

}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::solve(Mesh& mesh,
				    Function& f,
				    Function& v0,
				    Function& rho,
				    real E, real nu, real nuv,
				    BoundaryCondition& bc,
				    real k, real T)
{
  ElasticityUpdatedSolver solver(mesh, f, v0, rho, E, nu, nuv, bc, k, T);
  solver.solve();
}
//-----------------------------------------------------------------------------
