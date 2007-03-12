// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
// Modified by Anders Logg 2005-2006.
//
// First added:  2005
// Last changed: 2006-08-08

#include <iostream>
#include <dolfin/timing.h>
#include "dolfin/CNSSolver.h"
#include "dolfin/CNSmix2D.h"
#include "dolfin/CNSmix3D.h"
#include "dolfin/CNSResDensity.h"
#include "dolfin/CNSResMomentum.h"
#include "dolfin/CNSResEnergy.h"


using namespace dolfin;

//-----------------------------------------------------------------------------
CNSSolver::CNSSolver(Mesh& mesh, Function& f, Function& initial, BoundaryCondition& bc)
  : mesh(mesh), f(f), initial(initial), bc(bc) 
{
  // Declare parameters
  add("velocity file name", "velocity.pvd");
  add("pressure file name", "pressure.pvd");
  add("density file name", "density.pvd");
  add("momentum file name", "momentum.pvd");
  add("energy file name", "energy.pvd");
  add("res_den file name", "res_den.pvd");
  add("res_m file name", "res_m.pvd");

  add("nu_den file name", "nu_den.pvd");
}
//-----------------------------------------------------------------------------
void CNSSolver::solve()
{
  real T0 = 0.0;        // start time 
  real t  = 0.0;        // current time
  real T  = 1;        // final time
  //real nu = 1.0/3900.0; // viscosity 
  real nu = 0.0;
  real gamma = 1.5;

  // Set time step (proportional to the minimum cell diameter) 
  real hmin;
  GetMinimumCellSize(mesh, hmin);  
  //real k = 0.015*hmin; 
  real k = 0.25*hmin; 
  //real k = 0.5*hmin; 0.25*hmin; 

  //  Create matrix and vector
  Matrix A;
  Vector b;

  //  Get the number of space dimensions of the problem 
  
  int nsdim = mesh.topology().dim();     // nsdim: is number of space dimensions
  int nsd = nsdim + 2;                   // nsd:       is number of variables

  dolfin_info("Number of space dimensions: %d",nsdim);


  Vector x0(nsd*mesh.numVertices());      // x0:   solution from previous timestep
  Vector x(nsd*mesh.numVertices());       // x:    current solution
  Vector xold(nsd*mesh.numVertices());    // xold: save current solution 

  Vector px(mesh.numVertices());          // px:   pressure
  Vector ux(nsdim*mesh.numVertices());    // ux:   velocity
  Vector u0x(nsdim*mesh.numVertices());   // u0x:  velocity from previous timestep

  Vector rhox(mesh.numVertices());        // rhox: density
  Vector rho0x(mesh.numVertices());       // rho0x: density from previous timestep

  Vector mx(nsdim*mesh.numVertices());    // mx:   momentum
  Vector m0x(nsdim*mesh.numVertices());   // m0x:   momentum from previous timestep

  Vector ex(mesh.numVertices());          // ex:   energy
  Vector e0x(mesh.numVertices());         // e0x:   energy from previous timestep

  Vector vol_invx(mesh.numCells());       // vol_invx: needed for the computing strong residual
  Vector res_rhox(mesh.numCells());       // res_rhox: needed for storing the density residual
  Vector res_mx(nsdim * mesh.numCells());         // res_mx: needed for storing the momentum residual
  Vector res_ex(mesh.numCells());         // res_ex: needed for storing the energy residual

  Vector nu_rhox(mesh.numCells());        // res_rhox: needed for storing the density residual
  Vector nu_mx(mesh.numCells());          // res_mx: needed for storing the momentum residual
  Vector nu_ex(mesh.numCells());          // res_ex: needed for storing the energy residual

  Vector dvector(mesh.numCells());        // dvector: needed for the stabilazation term

  // Initialize the vectors:
  x0       = 0.0;
  x        = 0.0;
  xold     = 0.0;
  res_rhox = 0.0;
  res_mx   = 0.0;
  res_ex   = 0.0;
  nu_rhox  = 0.0;
  nu_mx    = 0.0;  
  nu_ex    = 0.0;

  //  Initialize vectors for the residuals of 
  //  the equations in fix point iteration
  Vector residual(nsd*mesh.numVertices());
  residual = 1.0e3;

  //  Initialize algebraic solvers 
  KrylovSolver solver_con(gmres, amg);
  KrylovSolver solver(gmres);

  //  Initialize stabilization parameters 
  Function delta(dvector);

  //  Create functions for the solution
  //  (needed for the initialization of the forms)
  Function w(x, mesh);               // current solution
  Function w0(x0, mesh);             // solution from previous time step
  Function wold(xold, mesh);             // solution from previous time step

  Function P(px, mesh);              // pressure 
  Function u(ux, mesh);              // velocity
  Function u0(u0x, mesh);            // velocity from previous

  Function rho(rhox, mesh);          // density 
  Function rho0(rho0x, mesh);        // density from previous timestep

  Function m(mx, mesh);              // momentum
  Function m0(m0x, mesh);            // momentum from previous timestep

  Function e(ex, mesh);              // energy
  Function e0(e0x, mesh);            // energy from previous timestep

  Function vol_inv(vol_invx, mesh);  // 1 / volume of element
  Function res_rho(res_rhox, mesh);  // residual for density
  Function res_m(res_mx, mesh);      // residual for momentum
  Function res_e(res_ex, mesh);      // residual for energy

  Function nu_rho(nu_rhox, mesh);    // residual for density
  Function nu_m(nu_mx, mesh);        // residual for momentum
  Function nu_e(nu_ex, mesh);        // residual for energy


  // Initialize the bilinear and linear forms for the system
  BilinearForm* a = 0;
  LinearForm* L = 0;

  //  Initialize the linear forms for the strong residual of density, momentum and energy
  LinearForm* Lres_rho = 0;
  LinearForm* Lres_m = 0;
  LinearForm* Lres_e = 0;

  if ( nsdim == 3 )
    {
//      a = new CNSmix3D::BilinearForm(u, nu_rho, nu_m, nu_e, k);
//      L = new CNSmix3D::LinearForm(u, delta, nu_rho, nu_m, nu_e, k);
      a = new CNSmix3D::BilinearForm(u, k, nu);
      L = new CNSmix3D::LinearForm(u, delta, k, nu);

    }
  else if ( nsdim == 2 )
    {
      a = new CNSmix2D::BilinearForm(u, delta, nu_rho, nu_m, nu_e, k);
      L = new CNSmix2D::LinearForm(w0, P, u, delta, nu_rho, nu_m, nu_e, k);

      Lres_rho = new CNSResDensity::LinearForm(u, rho, rho0, vol_inv, k);
      Lres_m   = new CNSResMomentum::LinearForm(P, u, m, m0, vol_inv, k);
      Lres_e   = new CNSResEnergy::LinearForm(P, u, e, e0, vol_inv, k);
    }
  else
    {
      dolfin_error("Navier-Stokes solver only implemented for 2 and 3 space dimensions.");
    }
  
  // Initialize the matrix A and vector b
  b.init(nsd*mesh.numVertices());
  A.init(nsd*mesh.numVertices(), nsd*mesh.numVertices());

  // Attaching discrete function data
  w0.attach(a->trial());
  w.attach(a->trial());

  res_rho.attach(nu_rho.element());
  res_m.attach(Lres_m->test());
  res_e.attach(nu_e.element());

  // Set initial data
  //finterpolate(initial, w, mesh);
  w.interpolate(initial);

  x0 = x;  

  // Initialize output files 
  File file_u(get("velocity file name"));   // file for saving velocity 
  File file_p(get("pressure file name"));   // file for saving pressure
  File file_rho(get("density file name"));  // file for saving pressure
  File file_m(get("momentum file name"));   // file for saving pressure
  File file_e(get("energy file name"));     // file for saving pressure
  File file_res(get("res_m file name"));     // file for saving pressure
  File file_nu(get("nu_den file name"));     // file for saving pressure
  
  // Compute velocity and pressure
  ComputeUP(mesh, w, P, u, gamma, nsdim);

  // Pick up the components from w0 and w
  ComputeRME(mesh, w0, rho0, m0, e0);
  ComputeRME(mesh, w, rho, m, e);
  
  // Compute the volume inverse of an element
  ComputeVolInv(mesh, vol_invx);


  // Save the initial condition to file
  //file_p << P;
  file_u << u;
  file_rho << rho;
  //file_m << m;
  //file_e << e;
  //file_res << res_m;
  //file_nu << nu_rho;
  
  
  // Synchronise functions and boundary conditions with time
  w.sync(t);
  bc.sync(t);
  
  // Compute stabilization parameters
  // ComputeStabilization(mesh,u,k,dvector);

  dolfin_info("Assembling matrix: compressible");

  // Assembling matrices 
  FEM::assemble(*Lres_rho, res_rhox, mesh);
  FEM::assemble(*Lres_m, res_mx, mesh);
  FEM::assemble(*Lres_e, res_ex, mesh);
  ComputeNu(mesh, res_rho, res_m, res_e, nu_rho, nu_m, nu_e);

  FEM::assemble(*a, A, mesh);

  // Initialize time-stepping parameters
  int time_step = 0;
  int sample = 0;
  int no_samples = 100;

  // Residual, tolerance and maxmimum number of fixed point iterations
  real scalar_residual;
  real rtol = 1.0e-2;
  int iteration;
  int max_iteration = 50;

  // Start time-stepping
  Progress prog("Time-stepping");
  while (t<T) 
  {

    time_step++;
    dolfin_info("Time step %d",time_step);

    // Set current solution to solution at previous time step 
    x0 = x;

    //u0 = u;

    // Compute velocity and pressure
    //ComputeUP(mesh, w, P, u, gamma, nsdim);

    // Compute stabilization parameters
    ComputeStabilization(mesh,u,k,dvector);
    // Initialize residual 
    scalar_residual = 2*rtol;
    iteration = 0;

    // Fix-point iteration for non-linear problem 
    while (scalar_residual > rtol && iteration < max_iteration){
      
      dolfin_info("Assemble vector: compressible");

      // Compute velocity and pressure
      ComputeUP(mesh, w, P, u, gamma, nsdim);

      // Assemble compressible vector 
      tic();
      
      FEM::assemble(*a, A, mesh);
      FEM::assemble(*L, b, mesh);

      dolfin_info("Assemble took %g seconds",toc());

      // Set boundary conditions 
      FEM::applyBC(A, b, mesh, a->trial(),bc);

      // Store solution of previous iteration
      xold = x;
            
      // Solve the linear system  
      dolfin_info("Solve linear system: compressible");
      tic();
      solver.solve(A, x, b);
      dolfin_info("Linear solve took %g seconds",toc());

      // Compute residual  
      residual = xold;
      residual -= x;
      
      // residual = 0.0;
      
      dolfin_info("Residual  : l2 norm = %e",residual.norm());
      scalar_residual = residual.norm();
      iteration++;
    }
    
    // Compute shock capturing parameters
    ComputeRME(mesh, w0, rho0, m0, e0);
    ComputeRME(mesh, w, rho, m, e);
    FEM::assemble(*Lres_rho, res_rhox, mesh);
    FEM::assemble(*Lres_m, res_mx, mesh);
    FEM::assemble(*Lres_e, res_ex, mesh);
    ComputeNu(mesh, res_rho, res_m, res_e, nu_rho, nu_m, nu_e);
    
    if(scalar_residual > rtol)
      dolfin_warning("CNS fixed point iteration did not converge"); 

    if ( (time_step == 1) || (t > (T-T0)*(real(sample)/real(no_samples))) ){
    //if ( true ){
      dolfin_info("save solution to file");

      //file_p << P;
      file_u << u;
      file_rho << rho;
      //file_m << m;
      //file_e << e;
      //file_res << res_m;
      //file_nu << nu_rho;
      

      sample++;
    }

    // Increase time with timestep
    t = t + k;

    // Update progress
    prog = t / T;
  }

  ComputeRME(mesh, w0, rho0, m0, e0);
  ComputeRME(mesh, w, rho, m, e);
  dolfin_info("save solution to file");
  //file_p << P;
  file_u << u;
  file_rho << rho;
  //file_m << m;
  //file_e << e;
  //file_res << res_rho;

  delete a;
  delete L;
  delete Lres_rho;
  delete Lres_m;
  delete Lres_e;

}
//-----------------------------------------------------------------------------
void CNSSolver::solve(Mesh& mesh, Function& f, Function& initial, BoundaryCondition& bc)
{
  CNSSolver solver(mesh, f, initial, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
void CNSSolver::ComputeCellSize(Mesh& mesh, Vector& hvector)
{
  // Compute cell size h
  hvector.init(mesh.numCells());	
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      hvector((*cell).index()) = (*cell).diameter();
    }
}
//-----------------------------------------------------------------------------
void CNSSolver::GetMinimumCellSize(Mesh& mesh, real& hmin)
{
  // Get minimum cell diameter
  hmin = 1.0e6;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      if ((*cell).diameter() < hmin) hmin = (*cell).diameter();
    }
}
//-----------------------------------------------------------------------------
void CNSSolver::ComputeVolInv(Mesh& mesh, Vector& vol_inv)
{
  // Compute cell size h
  vol_inv.init(mesh.numCells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      vol_inv((*cell).index()) = 1.0 / (*cell).volume();  // (*cell).diameter();
    }
}


//-----------------------------------------------------------------------------
void CNSSolver::ComputeStabilization(Mesh& mesh, Function& u, real k, 
				     Vector& dvector)
{
  // Compute least-squares stabilizing terms: 
  //
  // if  h/nu > 1 or ny < 10^-10
  //   d1 = C1 * ( 0.5 / sqrt( 1/k^2 + |U|^2/h^2 ) )   
  //   d2 = C2 * h 
  // else 
  //   d1 = C1 * h^2  
  //   d2 = C2 * h^2  

//   real C1 = 4.0;   
  real C1 = 4.0;   
  //  real C1 = 0.0; // * 1.0e0;   
 

  int N = mesh.numVertices();

  dvector.init(mesh.numCells());	
  //d2vector.init(mesh.numCells());	

  real normu; 

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    normu = 0.0;
    for (VertexIterator n(cell); !n.end(); ++n)
      {
	real csum = 0.0;
	for (uint i = 0; i < mesh.geometry().dim(); i++)
	  {
	    csum += sqr((u.vector())(i * N + (*n).index()));
	  }
	normu += sqrt(csum);
      }
    normu /= (*cell).numEntities(0);

    dvector((*cell).index()) =  C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(normu/(*cell).diameter()) ) );
    //dvector((*cell).index()) = C1 * sqr((*cell).diameter());
  }
}

//-----------------------------------------------------------------------------
void CNSSolver::ComputeNu(Mesh& mesh, Function& res_rho, Function& res_m, Function& res_e, 
			  Function& nu_rho, Function& nu_m, Function& nu_e)
{

  //int N = mesh.numVertices();
  int M = mesh.numCells();

  cout << "M: " << M << endl;
  cout << "res_m.size(): " << res_m.vector().size() << endl;
  cout << "res_m.vectordim(): " << res_m.vectordim() << endl;
  cout << "res_rho.size(): " << res_rho.vector().size() << endl;

  real normres = 0.0; 
  real h; 
 
  //real C = 1.0;
  real C = 1.0;
  
  // nu_rho
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      uint id = (*cell).index();

      normres = sqr(res_rho.vector()(id));		    
      normres = sqrt(normres);
      h = (*cell).diameter();
      nu_rho.vector()(id) = C * normres * h * h ;
      //nu_rho.vector()(id) = C * h ;
    }
  normres = 0.0; 
  // nu_m
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      uint id = (*cell).index();
      
      //      normres = sqr(res_m.vector()(0 * N + id)) + sqr(res_m.vector()(1 * N + id));
      normres = sqr(res_m.vector()(0 * M + id)) + sqr(res_m.vector()(1 * M + id));
      normres = sqrt(normres);
      h = (*cell).diameter();
      nu_m.vector()(id) = C * normres * h * h ;
      //nu_m.vector()(id) = C * h;
    }

  normres = 0.0; 
  // nu_e
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      uint id = (*cell).index();

      normres = sqr(res_rho.vector()(id));
      normres = sqrt(normres);
      h = (*cell).diameter();
      nu_e.vector()(id) = C * normres * h * h ;
      //nu_e.vector()(id) = C * h;
    }
  
}



//-----------------------------------------------------------------------------
void CNSSolver::SetInitialData(Function& rho0, Function& m0, Function& e0)
{
  // Function for setting initial velocity, 
  // given as a function of the coordinates (x,y,z).
  //
  // This function is only temporary: initial velocity 
  // should be possible to set in the main-file. 


//  real x,y,z;

  for (VertexIterator vi(mesh); !vi.end(); ++vi)
  {
    Vertex& v = *vi;
    int index = v.index();

    // Get coordinates of the vertex 
    //real x = v.point().x();
    //real y = v.point().y();
    //real z = v.point().z();
    
    // Specify the initial velocity using (x,y,z) 
    //
    // Example: 
    // 
    // xvel((*vertex).index()) = sin(x)*cos(y)*sqrt(z);
    
    //x((*vertex).index()) = 0.0;

    rho0.vector()(index) = 1.0;
    m0.vector()(index) = 0.0;
    e0.vector()(index) = 0.0;
  }

  rho0.vector() = 1.0;
}
//-----------------------------------------------------------------------------
// void CNSSolver::finterpolate(Function& f1, Function& f2,
// 			     Mesh& mesh)
// {
//   FiniteElement& element = f2.element();
//   Vector& x = f2.vector();

//   AffineMap map;

//   int *nodes = new int[element.spacedim()];
//   real *coefficients = new real[element.spacedim()];

//   for(CellIterator c(mesh); !c.end(); ++c)
//   {
//     Cell& cell = *c;

//     // Use DOLFIN's interpolation

//     map.update(cell);
//     f1.interpolate(coefficients, map, element);
//     element.nodemap(nodes, cell, mesh);

//     for(unsigned int i = 0; i < element.spacedim(); i++)
//     {
//       x(nodes[i]) = coefficients[i];
//     }
//   }

//   delete [] nodes;
//   delete [] coefficients;
// }
//-----------------------------------------------------------------------------
void CNSSolver::ComputeUP(Mesh& mesh, Function& w, Function& p, Function& u, real gamma, int nsdim)
{
  int N = mesh.numVertices();

  real rhoval, m0val, m1val, eval;

  //Vector& wx = w.vector();

  for (VertexIterator n(mesh); !n.end(); ++n)
    {
      int id = (*n).index();

      Point pp = (*n).point();
      
      rhoval = w.vector()(0 * N + id);
      m0val = w.vector()(1 * N + id);
      m1val = w.vector()(2 * N + id);
      eval = w.vector()(3 * N + id);
      
      u.vector().set(0 * N + id, m0val / rhoval);
      u.vector().set(1 * N + id, m1val / rhoval);

      p.vector().set(id, (gamma - 1.0) * (eval - rhoval * 0.5 * (sqr(u.vector().get(0 * N + id)) +
      								 sqr(u.vector().get(1 * N + id)))));
    }
}
//-----------------------------------------------------------------------------
void CNSSolver::ComputeRME(Mesh& mesh, Function& w, Function& rho, Function& m, Function& e)
{
  int N = mesh.numVertices();

  real rhoval, m0val, m1val, eval;

  //Vector& wx = w.vector();

  for (VertexIterator n(mesh); !n.end(); ++n)
    {
      int id = (*n).index();
      
      rhoval = w.vector()(0 * N + id);
      m0val = w.vector()(1 * N + id);
      m1val = w.vector()(2 * N + id);
      eval = w.vector()(3 * N + id);


      rho.vector().set(id, rhoval);
      m.vector().set(0 * N + id, m0val);
      m.vector().set(1 * N + id, m1val);
      e.vector().set(id, eval);
     
    }
}
