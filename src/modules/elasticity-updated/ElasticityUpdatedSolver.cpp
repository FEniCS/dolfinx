// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2004-2006.
// Modified by Garth N. Wells 2005.
//
// First added:  2005
// Last changed: 2006-02-20

//#include <iostream>
#include <sstream>
#include <iomanip>

#include "dolfin/timing.h"
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
    t(0.0), rtol(get("discrete tolerance")),
    maxiters(get("maximum iterations")), do_plasticity(false),
    yield(0.0),
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

  dotu_x1.init(Nv);
  dotu_x2.init(Nv);

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

  dotu_xsigma.init(Nsigma);

  dotu.init(2 * Nv + Nsigma);

  xjaumann1.init(Nsigma);

  xepsilon1.init(Nsigma);

  xsigmanorm.init(Nsigmanorm);

  stepresidual.init(Nv);

  // ODE indices

  dotu_x1_indices = new int[Nv];
  dotu_x2_indices = new int[Nv];
  dotu_xsigma_indices = new int[Nsigma];

  for(uint i = 0; i < Nv; i++)
  {
    dotu_x1_indices[i] = i;
    dotu_x2_indices[i] = Nv + i;
  }

  for(uint i = 0; i < Nsigma; i++)
  {
    dotu_xsigma_indices[i] = 2 * Nv + i;
  }

  // Set initial velocities

  AffineMap map;

  {
  int *nodes = new int[element1.spacedim()];
  real *coefficients = new real[element1.spacedim()];

  for(CellIterator c(&mesh); !c.end(); ++c)
  {
    Cell& cell = *c;

    // Use DOLFIN's interpolation

    map.update(cell);
    v0.interpolate(coefficients, map, element1);
    element1.nodemap(nodes, cell, mesh);

    for(uint i = 0; i < element1.spacedim(); i++)
    {
      x2_1(nodes[i]) = coefficients[i];
    }
  }

  delete [] nodes;
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
  int *nodes = new int[element2.spacedim()];
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    Cell& cell = *c;

    element2.nodemap(nodes, cell, mesh);

    real factor = 1.0 / cell.volume(); 

    for(uint i = 0; i < element2.spacedim(); i++)
      msigma(nodes[i]) = factor;
  }
  delete [] nodes;
  }

  // The mesh points are the initial values of u
  int offset = mesh.numVertices();
  for (VertexIterator n(&mesh); !n.end(); ++n)
  {
    Vertex& vertex = *n;
    int nid = vertex.id();

    x1_1(0 * offset + nid) = vertex.coord().x;
    x1_1(1 * offset + nid) = vertex.coord().y;
    x1_1(2 * offset + nid) = vertex.coord().z;
  }

  int dotu_x1offset = 0;
  
  ISCreateBlock(MPI_COMM_WORLD, Nv, 1, &dotu_x1offset, &dotu_x1is);
  VecScatterCreate(dotu_x1.vec(), PETSC_NULL, dotu.vec(), dotu_x1is,
		   &dotu_x1sc);

  int dotu_x2offset = Nv;

  ISCreateBlock(MPI_COMM_WORLD, Nv, 1, &dotu_x2offset, &dotu_x2is);
  VecScatterCreate(dotu_x2.vec(), PETSC_NULL, dotu.vec(), dotu_x2is,
		   &dotu_x2sc);

  int dotu_xsigmaoffset = 2 * Nv;

  ISCreateBlock(MPI_COMM_WORLD, Nsigma, 1, &dotu_xsigmaoffset, &dotu_xsigmais);
  VecScatterCreate(dotu_xsigma.vec(), PETSC_NULL, dotu.vec(), dotu_xsigmais,
		   &dotu_xsigmasc);

  // Initial values for ODE

  dotu_x1 = x1_1;
  dotu_x2 = x2_1;
  dotu_xsigma = xsigma1;

  // Gather values into dotu
  gather(dotu_x1, dotu, dotu_x1sc);
  gather(dotu_x2, dotu, dotu_x2sc);
  gather(dotu_xsigma, dotu, dotu_xsigmasc);
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
    x1_1.axpy(k, dotu_x1);

    // Time step method (dG(0)), velocity (also compute residual)

    stepresidual = x2_0;
    stepresidual.axpy(k, dotu_x2);
    stepresidual.axpy(-1, x2_1);
    x2_1 += stepresidual;

//     cout << "x2_1:" << endl;
//     x2_1.disp();

    
    cout << "stepresidual(j): " << stepresidual.norm(Vector::linf) << endl;

    // Time step method (dG(0)), stress

    xsigma1 = xsigma0;
    xsigma1.axpy(k, dotu_xsigma);


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
  
  File         file("elasticity.pvd");

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
//     std::ostringstream fileid, filename;
//     fileid.fill('0');
//     fileid.width(6);
    
//     fileid << counter;
    
//     filename << "mesh" << fileid.str() << ".xml.gz";
    
//     cout << "writing: " << filename.str() << " at t: " << t << endl;
    
//     std::string foo = filename.str();
//     const char *fname = foo.c_str();
    
//     File meshfile(fname);
    
//     meshfile << mesh;

  solutionfile << u1;
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
  fevals++;

  // Compute dotu_x1, dotu_x2 and dotu_xsigma based on x1_1, x2_1 and xsigma1
  // Ultimately compute dotu = f(u, t)

  // Update the mesh
  for (VertexIterator n(&mesh); !n.end(); ++n)
  {
    Vertex& vertex = *n;
    
    vertex.coord().x = u1(vertex, 0);
    vertex.coord().y = u1(vertex, 1);
    vertex.coord().z = u1(vertex, 2);
  }

  // Compute norm of stress (sigmanorm)
  if(do_plasticity)
  {
    {
      int *nodes = new int[element2.spacedim()];
      for (CellIterator c(mesh); !c.end(); ++c)
      {
	Cell& cell = *c;
	
	element2.nodemap(nodes, cell, mesh);
	
	real proj = 1;
	real norm = 0;
	for(uint i = 0; i < element2.spacedim(); i++)
	{
	  norm = std::max(norm, fabs(dotu_xsigma(nodes[i])));
	}
	
	if(norm > yield)
	{
	  cout << "sigmanorm(" << cell.id() << "): " << norm << endl;
	  proj = 1.0 / norm;
	}
	
	xsigmanorm(nodes[0]) = proj;
      }
      delete [] nodes;
    }
  }
  
  // xepsilon1 (needed for dotu_x2)

  xepsilon1 = dotu_xsigma;
  xepsilon1 *= 1.0 / lambda;

  // dotu_x1
  dotu_x1 = x2_1;


  // xsigma1
  dolfin_log(false);
  FEM::assemble(Lsigma, xsigmatmp1, mesh);
  dolfin_log(true);
  VecPointwiseMult(dotu_xsigma.vec(), xsigmatmp1.vec(), msigma.vec());
    
  dotu_xsigma.apply();
    
  // dotu_x2

  // Assemble v vector
  dolfin_log(false);
  FEM::assemble(Lv, xtmp1, mesh);
  FEM::applyBC(Dummy, xtmp1, mesh, element1, bc);

  VecPointwiseDivide(dotu_x2.vec(), xtmp1.vec(), m.vec());
  dotu_x2.apply();

  // Add contact forces
//   xtmp1.axpy(1, fcontact);
  dotu_x2.axpy(1, fcontact);

  dolfin_log(true);

  // Gather values into dotu
  gather(dotu_x1, dotu, dotu_x1sc);
  gather(dotu_x2, dotu, dotu_x2sc);
  gather(dotu_xsigma, dotu, dotu_xsigmasc);
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedSolver::gather(Vector& x1, Vector& x2, VecScatter& x1sc)
{
  VecScatterBegin(x1.vec(), x2.vec(), INSERT_VALUES, SCATTER_FORWARD,
		  x1sc);
  VecScatterEnd(x1.vec(), x2.vec(), INSERT_VALUES, SCATTER_FORWARD,
		x1sc);
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
   return solver.dotu(i);
}
//-----------------------------------------------------------------------------
void ElasticityUpdatedODE::f(const real u[], real t, real y[])
{
  // Copy values from ODE array
  fromArray(u, solver.x1_1, 0, solver.Nv);
  fromArray(u, solver.x2_1, solver.Nv, solver.Nv);
  fromArray(u, solver.xsigma1, 2 * solver.Nv, solver.Nsigma);

  solver.prepareiteration();

  // Compute solver RHS (puts result in Vector variables)
  solver.fu();

  // Copy values into ODE array
  toArray(y, solver.dotu, 0, 2 * solver.Nv + solver.Nsigma);
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
