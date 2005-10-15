// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2004-2005.
// Modified by Garth N. Wells 2005.
//
// First added:  2005
// Last changed: 2005-09-16

//#include <iostream>
#include <sstream>
#include <iomanip>

#include "dolfin/timeinfo.h"
#include "dolfin/ElasticityUpdatedSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ElasticityUpdatedSolver::ElasticityUpdatedSolver(Mesh& mesh, 
						 Function& f,
						 Function& v0,
						 Function& rho,
						 real& E, real& nu, real& nuv,
						 real& nuplast,
						 BoundaryCondition& bc,
						 real& k, real& T)
  : mesh(mesh), f(f), v0(v0), rho(rho), E(E), nu(nu), nuv(nuv),
    nuplast(nuplast), bc(bc), k(k),
    T(T), counter(0), lastsample(0),
    lambda(E * nu / ((1 + nu) * (1 - 2 * nu))),
    mu(E / (2 * (1 + nu))),
    t(0.0), rtol(1.0e-4), maxiters(10), do_plasticity(false), yield(0.0),
    savesamplefreq(33.0),
    fevals(0),
    ode(0),
    v1(x2_1, mesh, element1),
    u0(x1_0, mesh, element1),
    u1(x1_1, mesh, element1),
    sigma0(xsigma0, mesh, element2),
    sigma1(xsigma1, mesh, element2),
    epsilon1(xepsilon1, mesh, element2),
    sigmanorm(xsigmanorm, mesh, element3),
    Lv(f, sigma1, epsilon1, nuv),
    Lsigma(v1, sigma1, sigmanorm, lambda, mu, nuplast)
{
  cout << "nuv: " << nuv << endl;
  cout << "lambda: " << lambda << endl;
  cout << "mu: " << mu << endl;

  init();
  ode = new ElasticityUpdatedODE(*this);
  ts = new TimeStepper(*ode);
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::init()
{
  Matrix M;

  Nv = FEM::size(mesh, element1);
  Nsigma = FEM::size(mesh, element2);
  Nsigmanorm = FEM::size(mesh, element3);

  x1_0.init(Nv);
  x1_1.init(Nv);
  x2_0.init(Nv);
  x2_1.init(Nv);

  x1ode.init(Nv);
  x2ode.init(Nv);

  xtmp1.init(Nv);
  xtmp2.init(Nv);

  fcontact.init(Nv);

  Dummy.init(Nv, Nv);

  for(uint i = 0; i < Nv; i++)
  {
    Dummy(i, i) = 1.0;
  }


  msigma.init(Nsigma);

  xsigmatmp1.init(Nsigma);
  xsigmatmp2.init(Nsigma);

  xsigma0.init(Nsigma);
  xsigma1.init(Nsigma);

  xsigmaode.init(Nsigma);

  dotu.init(2 * Nv + Nsigma);

  xjaumann1.init(Nsigma);

  xepsilon1.init(Nsigma);

  xsigmanorm.init(Nsigmanorm);

  stepresidual.init(Nv);

  // ODE indices

  x1ode_indices = new int[Nv];
  x2ode_indices = new int[Nv];
  xsigmaode_indices = new int[Nsigma];

  for(uint i = 0; i < Nv; i++)
  {
    x1ode_indices[i] = i;
    x2ode_indices[i] = Nv + i;
  }

  for(uint i = 0; i < Nsigma; i++)
  {
    xsigmaode_indices[i] = 2 * Nv + i;
  }

  uode = new real[2 * Nv + Nsigma];
  yode = new real[2 * Nv + Nsigma];


  // Set initial velocities

  AffineMap map;
  v0.set(element1);

  {
  int *dofs = new int[element1.spacedim()];
  real *coefficients = new real[element1.spacedim()];
  for(CellIterator c(&mesh); !c.end(); ++c)
  {
    Cell& cell = *c;

    // Use DOLFIN's interpolation

    map.update(cell);
    v0.interpolate(coefficients, map);
    element1.dofmap(dofs, cell, mesh);

    for(uint i = 0; i < element1.spacedim(); i++)
      x2_1(dofs[i]) = coefficients[i];
  }
  delete [] dofs;
  delete [] coefficients;
  }

  FEM::applyBC(Dummy, x2_1, mesh, element1, bc);

  cout << "x2_1:" << endl;
  x2_1.disp();

  // Set initial stress

  xsigma1 = 0.0;

  cout << "xsigma1:" << endl;
  xsigma1.disp();

  xsigmanorm = 1.0;

  ElasticityUpdatedMass::BilinearForm amass(rho);

  // Assemble mass matrix
  FEM::assemble(amass, M, mesh);

  // Lump mass matrix
  FEM::lump(M, m);

  cout << "m:" << endl;
  m.disp();


  // Compute mass vector (sigma)
  {
  int *dofs = new int[element2.spacedim()];
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    Cell& cell = *c;

    element2.dofmap(dofs, cell, mesh);

    real factor = 1.0 / cell.volume(); 

    for(uint i = 0; i < element2.spacedim(); i++)
      msigma(dofs[i]) = factor;
  }
  delete [] dofs;
  }

  // The mesh points are the initial values of u
  int offset = mesh.noNodes();
  for (NodeIterator n(&mesh); !n.end(); ++n)
  {
    Node& node = *n;
    int nid = node.id();

    x1_1(0 * offset + nid) = node.coord().x;
    x1_1(1 * offset + nid) = node.coord().y;
    x1_1(2 * offset + nid) = node.coord().z;
  }

  int x1odeoffset = 0;
  
  ISCreateBlock(MPI_COMM_WORLD, Nv, 1, &x1odeoffset, &x1odeis);
  VecScatterCreate(x1ode.vec(), PETSC_NULL, dotu.vec(), x1odeis,
		   &x1odesc);

  int x2odeoffset = Nv;

  ISCreateBlock(MPI_COMM_WORLD, Nv, 1, &x2odeoffset, &x2odeis);
  VecScatterCreate(x2ode.vec(), PETSC_NULL, dotu.vec(), x2odeis,
		   &x2odesc);

  int xsigmaodeoffset = 2 * Nv;

  ISCreateBlock(MPI_COMM_WORLD, Nsigma, 1, &xsigmaodeoffset, &xsigmaodeis);
  VecScatterCreate(xsigmaode.vec(), PETSC_NULL, dotu.vec(), xsigmaodeis,
		   &xsigmaodesc);

  // Initial values for ODE

  x1ode = x1_1;
  x2ode = x2_1;
  xsigmaode = xsigma1;

  gather();
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::preparestep()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::prepareiteration()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::step()
{
  // Make time step

  t += k;
  cout << "t: " << t << endl;

  ts->step();
//   oldstep();
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::oldstep()
{
  x1_0 = x1_1;
  x2_0 = x2_1;
  xsigma0 = xsigma1;

  
  for(int iter = 0; iter < maxiters; iter++)
  {
    prepareiteration();

    // Compute RHS of ODE system
    fu();


    // Time step method (dG(0)), position

    x1_1 = x1_0;
    x1_1.axpy(k, x1ode);

    // Time step method (dG(0)), velocity (also compute residual)

    stepresidual = x2_0;
    stepresidual.axpy(k, x2ode);
    stepresidual.axpy(-1, x2_1);
    x2_1 += stepresidual;

//     cout << "x2_1:" << endl;
//     x2_1.disp();

    
    cout << "stepresidual(j): " << stepresidual.norm(Vector::linf) << endl;

    // Time step method (dG(0)), stress

    xsigma1 = xsigma0;
    xsigma1.axpy(k, xsigmaode);


//     cout << "x2_1:" << endl;
//     x2_1.disp();

    if(stepresidual.norm(Vector::linf) <= rtol && iter >= 0)
    {
      cout << "converged" << endl;
      break;
    }
    else if(iter == maxiters - 1)
    {
      cout << "did not converge" << endl;
    }
  }
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::solve()
{
  cout << "lambda: " << lambda << endl;
  cout << "mu: " << mu << endl;
  
  File         file("elasticity.m");

  // Synchronize f with time t
  f.sync(t);
  
  // Save the solution
  condsave(mesh, file, t);

  // Start a progress session
  Progress p("Time-stepping");
  
  // Start time-stepping
  while ( true && t < T ) {
  
    preparestep();
    step();
    
    // Save the solution
    condsave(mesh, file, t);

    // Benchmark
//     FEM::assemble(Lsigma0, xsigma0_1, mesh);
    
    // Update progress
    p = t / T;
  }

  cout << "total fevals: " << fevals << endl;
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::save(Mesh& mesh, File& solutionfile, real t)
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

}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::condsave(Mesh& mesh, File& solutionfile, real t)
{
//   real samplefreq = 1.0 / 33.0;
  real sampleperiod = 1.0 / savesamplefreq;

  while(lastsample + sampleperiod < t || t == 0.0)
  {
    save(mesh, solutionfile, t);

    counter++;

    lastsample = std::min(t, lastsample + sampleperiod);
    cout << "lastsample: " << lastsample << " t: " << t << endl;

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
				    real& E, real& nu, real& nuv,
				    real& nuplast,
				    BoundaryCondition& bc,
				    real& k, real& T)
{
  ElasticityUpdatedSolver solver(mesh, f, v0, rho, E, nu, nuv, nuplast,
				 bc, k, T);
  solver.solve();
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::fu()
{
  cout << "fu()" << endl;
  fevals++;

  // Compute x1ode, x2ode and xsigmaode based on x1_1, x2_1 and xsigma1
  // Ultimately compute dotu = f(u, t)

  // Update the mesh
  for (NodeIterator n(&mesh); !n.end(); ++n)
  {
    Node& node = *n;
    
    node.coord().x = u1(node, 0);
    node.coord().y = u1(node, 1);
    node.coord().z = u1(node, 2);
  }

  // Compute norm of stress (sigmanorm)
  if(do_plasticity)
  {
    {
      int *dofs = new int[element2.spacedim()];
      for (CellIterator c(mesh); !c.end(); ++c)
      {
	Cell& cell = *c;
	
	element2.dofmap(dofs, cell, mesh);
	
	real proj = 1;
	real norm = 0;
	for(uint i = 0; i < element2.spacedim(); i++)
	{
	  norm = std::max(norm, fabs(xsigmaode(dofs[i])));
	}
	
	if(norm > yield)
	{
	  cout << "sigmanorm(" << cell.id() << "): " << norm << endl;
	  proj = 1.0 / norm;
	}
	
	xsigmanorm(dofs[0]) = proj;
      }
      delete [] dofs;
    }
  }
  
  // xepsilon1 (needed for x2ode)

  xepsilon1 = xsigmaode;
  xepsilon1 *= 1.0 / lambda;

  // x1ode
  x1ode = x2_1;


  // xsigma1
  FEM::assemble(Lsigma, xsigmatmp1, mesh);
  VecPointwiseMult(xsigmaode.vec(), xsigmatmp1.vec(), msigma.vec());
    
  xsigmaode.apply();
    
  // x2ode

  // Assemble v vector
  FEM::assemble(Lv, xtmp1, mesh);
  
  // Add contact forces
  xtmp1.axpy(1, fcontact);
  
  FEM::applyBC(Dummy, xtmp1, mesh, element1, bc);
  VecPointwiseDivide(x2ode.vec(), xtmp1.vec(), m.vec());

  x2ode.apply();

  gather();
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::gather()
{
  // Gather values into dotu

//   VecScatter x1odesc;
//   IS x1odeis;
//   int x1odeoffset = 0;

//   ISCreateBlock(MPI_COMM_WORLD, Nv, 1, &x1odeoffset, &x1odeis);
//   VecScatterCreate(x1ode.vec(), PETSC_NULL, dotu.vec(), x1odeis,
// 		   &x1odesc);
  VecScatterBegin(x1ode.vec(), dotu.vec(), INSERT_VALUES, SCATTER_FORWARD,
		  x1odesc);
  VecScatterEnd(x1ode.vec(), dotu.vec(), INSERT_VALUES, SCATTER_FORWARD,
		x1odesc);
  

//   VecScatter x2odesc;
//   IS x2odeis;
//   int x2odeoffset = Nv;

//   ISCreateBlock(MPI_COMM_WORLD, Nv, 1, &x2odeoffset, &x2odeis);
//   VecScatterCreate(x2ode.vec(), PETSC_NULL, dotu.vec(), x2odeis,
// 		   &x2odesc);
  VecScatterBegin(x2ode.vec(), dotu.vec(), INSERT_VALUES, SCATTER_FORWARD,
		  x2odesc);
  VecScatterEnd(x2ode.vec(), dotu.vec(), INSERT_VALUES, SCATTER_FORWARD,
		x2odesc);


//   VecScatter xsigmaodesc;
//   IS xsigmaodeis;
//   int xsigmaodeoffset = 2 * Nv;

//   ISCreateBlock(MPI_COMM_WORLD, Nsigma, 1, &xsigmaodeoffset, &xsigmaodeis);
//   VecScatterCreate(xsigmaode.vec(), PETSC_NULL, dotu.vec(), xsigmaodeis,
// 		   &xsigmaodesc);
  VecScatterBegin(xsigmaode.vec(), dotu.vec(), INSERT_VALUES, SCATTER_FORWARD,
		  xsigmaodesc);
  VecScatterEnd(xsigmaode.vec(), dotu.vec(), INSERT_VALUES, SCATTER_FORWARD,
		xsigmaodesc);


//   cout << "dotu: " << endl;
//   dotu.disp();

}
//-----------------------------------------------------------------------------
ElasticityUpdatedODE::ElasticityUpdatedODE(ElasticityUpdatedSolver& solver) :
  ODE(1, 1.0), solver(solver)
{
  T = solver.T;
  N = 2 * solver.Nv + solver.Nsigma;
}
//-----------------------------------------------------------------------------
real ElasticityUpdatedODE::u0(unsigned int i)
{
//   if(i < solver.Nv)
//   {
//     return solver.x1_1(i);
//   }
//   else if(i >= solver.Nv && i < 2 * solver.Nv)
//   {
//     return solver.x2_1(i - solver.Nv);
//   }
//   else if(i >= 2 * solver.Nv && i < 2 * solver.Nv + solver.Nsigma)
//   {
//     return solver.xsigma1(i - 2 * solver.Nv);
//   }
//   else
//   {
//     dolfin_error("ElasticityUpdatedODE::u0(): out of bounds");
//     return 0.0;
//   }
  
   return solver.dotu(i);
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedODE::f(const real u[], real t, real y[])
{
  // Copy values from ODE array

  fromArray(u, solver.x1_1, 0, solver.Nv);
  fromArray(u, solver.x2_1, solver.Nv, solver.Nv);
  fromArray(u, solver.xsigma1, 2 * solver.Nv, solver.Nsigma);

  // Compute solver RHS (puts result in Vector variables)
  solver.fu();

  // Copy values into ODE array

//   toArray(y, solver.dotu, 0);
  toArray(y, solver.x1ode, 0, solver.Nv);
  toArray(y, solver.x2ode, solver.Nv, solver.Nv);
  toArray(y, solver.xsigmaode, 2 * solver.Nv, solver.Nsigma);
}
//-----------------------------------------------------------------------------
bool ElasticityUpdatedODE::update(const real u[], real t, bool end)
{
  fromArray(u, solver.x1_1, 0, solver.Nv);
  fromArray(u, solver.x2_1, solver.Nv, solver.Nv);
  fromArray(u, solver.xsigma1, 2 * solver.Nv, solver.Nsigma);

  return true;
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedODE::fromArray(const real u[], Vector& x, uint offset,
				     uint size)
{
  // Workaround to interface Vector and arrays

  real* vals = 0;
  vals = x.array();
  for(uint i = 0; i < size; i++)
  {
    vals[i] = u[i + offset];
  }
  x.restore(vals);

}
//-----------------------------------------------------------------------------
void ElasticityUpdatedODE::toArray(real y[], Vector& x, uint offset, uint size)
{
  // Workaround to interface Vector and arrays

  real* vals = 0;
  vals = x.array();
  for(uint i = 0; i < size; i++)
  {
    y[offset + i] = vals[i];
  }
  x.restore(vals);
  
}
//-----------------------------------------------------------------------------
