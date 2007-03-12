// Copyright (C) 2005 Johan Hoffman.
// LiceALEd under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
// Modified by Anders Logg 2005-2006.
//
// First added:  2005
// Last changed: 2006-08-08

#include <dolfin/timing.h>
#include <dolfin/ALESolver.h>
#include <dolfin/ALEMomentum3D.h>
#include <dolfin/ALEMomentum2D.h>
#include <dolfin/ALEContinuity3D.h>
#include <dolfin/ALEContinuity2D.h>

#include <dolfin/ALEBoundaryCondition.h>
#include <dolfin/ALEFunction.h>


using namespace dolfin;

// ALE: This class combines the mesh smooting and the 
// external force function to calculate the discrete
// mesh velocity function w.
class ALETools
{
public:

  ALETools(Mesh& mesh, Function& w, ALEFunction& e, real& k)
    : mesh(mesh), w(w), e(e), mvel(w.vector()), nsd(mesh.topology().dim()), k(k)
  { 


    boundary = new BoundaryMesh(mesh, vertex_map, cell_map);
    
    vertex_is_interior.init(mesh, 0);

    for (VertexIterator v(mesh); ! v.end(); ++v) {  
      
      vertex_is_interior.set(v->index(), true);
    }

    for (VertexIterator bv(*boundary); !bv.end(); ++bv) {  
      
      vertex_is_interior.set(vertex_map(*bv), false);
    }
  }
  //-----------------------------------------------------------------------------
  ~ALETools() 
  {
  }
  //-----------------------------------------------------------------------------
  // execute external function, smooth the mesh and from this
  // update the mesh velocity.
  void updateMesh()
  {
    mvel = 0.0;
    
    touchBoundary();  // let the external function touch the boundary
    
    smoothMesh();     // smooth the mesh
    
    mvel /= k;
  }
  // add more things, maybe.

private:

  bool isInterior(Vertex& vertex)
  {
    return (vertex_is_interior(vertex));
  }
  //-----------------------------------------------------------------------------
  void smoothMesh(void) 
  {
    unsigned int ndx;
    
    for (VertexIterator v(mesh); ! v.end(); ++v) {  
      
      if (!isInterior(*v)) continue;
      
      for (ndx = 0; ndx < nsd; ndx++) {
	
	real mass = 1;
	
	// move boundary point
	old_coord[ndx] = v->coordinates()[ndx]; 
	
	// iterate over the neighboring vertices
	VertexIterator vn(*v); 

	v->coordinates()[ndx] = vn->coordinates()[ndx];
	
	for (++vn; !vn.end(); ++vn) {
	  v->coordinates()[ndx] += vn->coordinates()[ndx];
	  mass += 1;
	}
	
	// divide by the number of neighbors
	v->coordinates()[ndx] /= mass;
	
	// mesh velocity contribution
	mvel.set(mvel_cpt(v->index(), ndx), mvel.get(mvel_cpt(v->index(), ndx)) + (v->coordinates()[ndx] - old_coord[ndx]));
      }	
    }
  }
  //-----------------------------------------------------------------------------
  void touchBoundary()
  {   

    unsigned int ndx;

    for (VertexIterator boundary_vertex(*boundary); !boundary_vertex.end(); ++boundary_vertex) {  
      
      Vertex v(mesh, vertex_map(*boundary_vertex));
  
      for (ndx = 0; ndx < nsd; ndx++) {
	  
	// move boundary point
	mesh.geometry().x(vertex_map(*boundary_vertex), ndx) 
	  += k * e.eval(v.point(), boundary_vertex->point(), ndx);
	
	// add mesh velocity contribution
	mvel.set(mvel_cpt(vertex_map(*boundary_vertex), ndx), 
		 mvel.get(mvel_cpt(vertex_map(*boundary_vertex), ndx)) + 
		 k * e.eval(v.point(), boundary_vertex->point(), ndx));
      }    
    }
  }
  //-----------------------------------------------------------------------------
  unsigned int mvel_cpt(unsigned int j, unsigned int i)
  {
    return (i + nsd*j);
  }

  Mesh&         mesh;
  BoundaryMesh* boundary;
  Function&     w;
  ALEFunction&  e;
  Vector&       mvel;
  unsigned int  nsd;
  real&         k;

  MeshFunction<bool>         vertex_is_interior;
  MeshFunction<unsigned int> vertex_map;
  MeshFunction<unsigned int> cell_map;
  

  real old_coord[3];
};
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
ALESolver::ALESolver(Mesh& mesh, 
		     Function& f, 
		     ALEBoundaryCondition& bc_mom, 
		     ALEBoundaryCondition& bc_con, 
		     ALEFunction& e)
  : mesh(mesh), f(f), bc_mom(bc_mom), bc_con(bc_con), e(e)
{
  // Declare parameters
  add("velocity file name", "velocity.pvd");
  add("pressure file name", "pressure.pvd");
}
//-----------------------------------------------------------------------------
void ALESolver::solve()
{
  real T0 = 0.0;        // start time 
  real t  = 0.0;        // current time
  real T  = 8.0;        // final time
  real nu = 1.0/3900.0; // viscosity 

  // Set time step (proportional to the minimum cell diameter) 
  real hmin;
  GetMinimumCellSize(mesh, hmin);  
  real k = 0.25*hmin; 
  // Create matrices and vectors 
  Matrix Amom, Acon;
  Vector bmom, bcon;

  // Get the number of space dimensions of the problem 
  int nsd = mesh.topology().dim();

  dolfin_info("Number of space dimensions: %d",nsd);

  // Initialize vectors for velocity and pressure 
  // x0vel: velocity from previous time step 
  // xcvel: linearized velocity 
  // xvel:  current velocity 
  // pvel:  current pressure 
  Vector x0vel(nsd*mesh.numVertices());
  Vector xcvel(nsd*mesh.numVertices());
  Vector xvel(nsd*mesh.numVertices());
  Vector xpre(mesh.numVertices());
  x0vel = 0.0;
  xcvel = 0.0;
  xvel = 0.0;
  xpre = 0.0;

  // ALE: Mesh velocity vector ========================================
  Vector mvel(nsd*mesh.numVertices());
  mvel = 0.0;
  // ==================================================================

  // Set initial velocity
  SetInitialVelocity(xvel);

  // Initialize vectors for the residuals of 
  // the momentum and continuity equations  
  Vector residual_mom(nsd*mesh.numVertices());
  Vector residual_con(mesh.numVertices());
  residual_mom = 1.0e3;
  residual_con = 1.0e3;

  // Initialize algebraic solvers 
  KrylovSolver solver_con(gmres, amg);
  KrylovSolver solver_mom(gmres);
 
  // Create functions for the velocity and pressure 
  // (needed for the initialization of the forms)
  Function u0(x0vel, mesh); // velocity from previous time step 
  Function uc(xcvel, mesh); // velocity linearized convection 
  Function p(xpre, mesh);   // current pressure

  // ALE: Move mesh velocity function =================================
  Function w(mvel, mesh);
  // ==================================================================

  
  // Initialize stabilization parameters 
  Vector d1vector, d2vector;
  Function delta1(d1vector), delta2(d2vector);

  // Initialize the bilinear and linear forms
  BilinearForm* amom = 0;
  BilinearForm* acon = 0;
  LinearForm* Lmom = 0;
  LinearForm* Lcon = 0;

  if ( nsd == 3 ) {

    // ALE: some need w = mesh_velocity ===============================
    amom = new ALEMomentum3D::BilinearForm(uc,delta1,delta2,w,k,nu);
    Lmom = new ALEMomentum3D::LinearForm(uc,u0,f,p,delta1,delta2,w,k,nu);
    acon = new ALEContinuity3D::BilinearForm(delta1);
    Lcon = new ALEContinuity3D::LinearForm(uc,f,delta1);
  } 
  else if ( nsd == 2 ) {

    // ALE: some need w = mesh_velocity ===============================
    amom = new ALEMomentum2D::BilinearForm(uc,delta1,delta2,w,k,nu);
    Lmom = new ALEMomentum2D::LinearForm(uc,u0,f,p,delta1,delta2,w,k,nu);
    acon = new ALEContinuity2D::BilinearForm(delta1);
    Lcon = new ALEContinuity2D::LinearForm(uc,f,delta1);
  }
  else
  {
    dolfin_error("Navier-Stokes solver only implemented for 2 and 3 space dimensions.");
  }

  // Create function for velocity 
  // (must be done after initialization of forms)
  Function u(xvel, mesh, uc.element());   // current velocity

  // Synchronise functions and boundary conditions with time
  u.sync(t);
  p.sync(t);
  bc_con.sync(t);
  bc_mom.sync(t);

  // ALE: =============================================================
  w.sync(t);   // mesh velocity function
  e.sync(t);   // external force function
  // ==================================================================
    
  // Initialize output files 
  File file_u(get("velocity file name"));  // file for saving velocity 
  File file_p(get("pressure file name"));  // file for saving pressure

  // Compute stabilization parameters
  ComputeStabilization(mesh,u0,nu,k,d1vector,d2vector);

  dolfin_info("Assembling matrix: continuity");

  // Assembling matrices 
  FEM::assemble(*amom, Amom, mesh);
  FEM::assemble(*acon, Acon, mesh);

  // Initialize time-stepping parameters
  int time_step = 0;
  int sample = 0;
  int no_samples = 100;

  // Residual, tolerance and maxmimum number of fixed point iterations
  real residual;
  real rtol = 1.0e-2;
  int iteration;
  int max_iteration = 50;

  // ALE: manipulation tools ==========================================
  ALETools ale(mesh, w, e, k);

  // passes boundary velocity function to bc
  bc_mom.setBoundaryVelocity(e);
  // ==================================================================

  // Start time-stepping
  Progress prog("Time-stepping");

  while (t<T) 
  {
		
    time_step++;
    dolfin_info("Time step %d",time_step);

    // Set current velocity to velocity at previous time step 
    x0vel = xvel;

    // Compute stabilization parameters
    ComputeStabilization(mesh,u0,nu,k,d1vector,d2vector);

    // Initialize residual 
    residual = 2*rtol;
    iteration = 0;
		
    // ALE: move continuity assembly inside ===========================
    // time incr. since mesh changes.
    FEM::assemble(*acon, Acon, mesh);
    // ================================================================
		
    // Fix-point iteration for non-linear problem 
    while (residual > rtol && iteration < max_iteration){
      
      dolfin_info("Assemble vector: continuity");

      // Assemble continuity vector 
      FEM::assemble(*Lcon, bcon, mesh);

      // Set boundary conditions for continuity equation 
      FEM::applyBC(Acon, bcon, mesh, acon->trial(), bc_con);

      // ALE: =========================================================
      bc_con.endRecording();
      // ==============================================================
			
      dolfin_info("Solve linear system: continuity");

      // Solve the linear system for the continuity equation 
      tic();
      solver_con.solve(Acon, xpre, bcon);
      dolfin_info("Linear solve took %g seconds",toc());

      dolfin_info("Assemble vector: momentum");

      // ALE: =========================================================
      FEM::assemble(*amom, Amom, mesh);
      // ==============================================================
			
      // Assemble momentum vector 
      tic();
      FEM::assemble(*Lmom, bmom, mesh);
      dolfin_info("Assemble took %g seconds",toc());
			
      // Set boundary conditions for the momentum equation 
      FEM::applyBC(Amom, bmom, mesh, amom->trial(), bc_mom);

      // ALE: =========================================================
      bc_mom.endRecording();
      // ==============================================================
      
      dolfin_info("Solve linear system: momentum");
			
      // Solve the linear system for the momentum equation 
      tic();
      solver_mom.solve(Amom, xvel, bmom);
      dolfin_info("Linear solve took %g seconds",toc());
      
      // Set linearized velocity to current velocity 
      xcvel = xvel;
    
      dolfin_info("Assemble matrix and vector: momentum");
      FEM::assemble(*amom, *Lmom, Amom, bmom, mesh, bc_mom);
      FEM::applyBC(Amom, bmom, mesh, amom->trial(),bc_mom);

      // Compute residual for momentum equation 
      Amom.mult(xvel,residual_mom);
      residual_mom -= bmom;
      
      dolfin_info("Assemble vector: continuity");
      FEM::assemble(*Lcon, bcon, mesh);
      FEM::applyBC(Acon, bcon, mesh, acon->trial(),bc_con);

      // Compute residual for continuity equation 
      Acon.mult(xpre,residual_con);
      residual_con -= bcon;
      
      dolfin_info("Momentum residual  : l2 norm = %e",residual_mom.norm());
      dolfin_info("continuity residual: l2 norm = %e",residual_con.norm());
      dolfin_info("Total ALE residual : l2 norm = %e",sqrt(sqr(residual_mom.norm()) + sqr(residual_con.norm())));

      residual = sqrt(sqr(residual_mom.norm()) + sqr(residual_con.norm()));
      iteration++;
    }

    // ALE: Apply external boundary force and smooth mesh =============
    // and reconstruct the mesh velocity function. 
    ale.updateMesh();
    // ================================================================
    
    if(residual > rtol)
      dolfin_warning("ALE fixed point iteration did not converge"); 
		
    if ( (time_step == 1) || (t > (T-T0)*(real(sample)/real(no_samples))) ) {
      dolfin_info("save solution to file");
      file_p << p;
      file_u << u;
      sample++;
    }
    
    // Increase time with timestep
    t = t + k;

    // Update progress
    prog = t / T;
  }

  dolfin_info("save solution to file");
  file_p << p;
  file_u << u;

  delete amom;
  delete Lmom;
  delete acon;
  delete Lcon;

}
//-----------------------------------------------------------------------------
void ALESolver::solve(Mesh& mesh, 
		      Function& f, 
		      ALEBoundaryCondition& bc_mom, 
		      ALEBoundaryCondition& bc_con, 
		      ALEFunction& e)
{
  ALESolver solver(mesh, f, bc_mom, bc_con, e);
  solver.solve();
}
//-----------------------------------------------------------------------------
void ALESolver::ComputeCellSize(Mesh& mesh, Vector& hvector)
{
  // Compute cell size h
  hvector.init(mesh.numCells());	
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      hvector((*cell).index()) = (*cell).diameter();
    }
}
//-----------------------------------------------------------------------------
void ALESolver::GetMinimumCellSize(Mesh& mesh, real& hmin)
{
  // Get minimum cell diameter
  hmin = 1.0e6;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      if ((*cell).diameter() < hmin) hmin = (*cell).diameter();
    }
}
//-----------------------------------------------------------------------------
void ALESolver::ComputeStabilization(Mesh& mesh, Function& w, real nu, real k, 
				     Vector& d1vector, Vector& d2vector)
{
  // Compute least-squares stabilizing terms: 
  //
  // if  h/nu > 1 or ny < 10^-10
  //   d1 = C1 * ( 0.5 / sqrt( 1/k^2 + |U|^2/h^2 ) )   
  //   d2 = C2 * h 
  // else 
  //   d1 = C1 * h^2  
  //   d2 = C2 * h^2  

  real C1 = 4.0;   
  real C2 = 2.0;   

  d1vector.init(mesh.numCells());	
  d2vector.init(mesh.numCells());	

  real normw; 

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    normw = 0.0;
    for (VertexIterator n(cell); !n.end(); ++n)
      normw += sqrt( sqr((w.vector())((*n).index()*2)) + sqr((w.vector())((*n).index()*2+1)) );
    normw /= (*cell).numEntities(0);
    if ( (((*cell).diameter()/nu) > 1.0) || (nu < 1.0e-10) ){
      d1vector((*cell).index()) = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(normw/(*cell).diameter()) ) );
      d2vector((*cell).index()) = C2 * (*cell).diameter();
    } else{
      d1vector((*cell).index()) = C1 * sqr((*cell).diameter());
      d2vector((*cell).index()) = C2 * sqr((*cell).diameter());
    }	
  }
}
//-----------------------------------------------------------------------------
void ALESolver::SetInitialVelocity(Vector& xvel)
{
  // Function for setting initial velocity, 
  // given as a function of the coordinates (x,y,z).
  //
  // This function is only temporary: initial velocity 
  // should be possible to set in the main-file. 


//  real x,y,z;

  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    // Get coordinates of the vertex 
//    real x = (*vertex).point().x();
//    real y = (*vertex).point().y;
//    real z = (*vertex).point().z;
    
    // Specify the initial velocity using (x,y,z) 
    //
    // Example: 
    // 
    // xvel((*vertex).index()) = sin(x)*cos(y)*sqrt(z);
    
    xvel((*vertex).index()) = 0.0;
  }
}
//-----------------------------------------------------------------------------
