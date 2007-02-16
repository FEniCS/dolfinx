// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2006-05-07


// PETSc friendly version

#ifndef __FSI_SOLVER_H
#define __FSI_SOLVER_H

#include <dolfin/Solver.h>

#include <dolfin/ALEBoundaryCondition.h>
#include <dolfin/ALEFunction.h>

#include <dolfin/BoundaryMesh.h>

namespace dolfin
{
  /// Fluid-Structure Interaction Solver
  class FSISolver 
  {
  public:
    
    /// Create the FSI solver
    FSISolver(Mesh& mesh, 
	      Function& f, 
	      ALEBoundaryCondition& bc_mom, 
	      ALEBoundaryCondition& bc_con, 
	      Function& bisect, 
	      real& rhof, 
	      real& rhos, 
	      real& E, 
	      real& elnu, 
	      real& nu,
	      real& T,
	      real& k);

    /// FSI solver destructor
    ~FSISolver();

    /// Solve FSI equations
    void solve();

    /// Solve FSI equations (static version)
    static void solve(Mesh& mesh, 
		      Function& f, 
		      ALEBoundaryCondition& bc_mom, 
		      ALEBoundaryCondition& bc_con, 
		      Function& bisect, 
		      real& rhof, 
		      real& rhos, 
		      real& E, 
		      real& elnu,
		      real& nu,
		      real& T,
		      real& k);

    /// Compute cell diameter
    void ComputeCellSize(Mesh& mesh, Vector& hvector);
      
    /// Get minimum cell diameter
    void GetMinimumCellSize(Mesh& mesh, real& hmin);

    /// Compute stabilization 
    void ComputeStabilization(Mesh& mesh, 
			      Function& w, 
			      real nu, 
			      real k, 
			      Vector& d1vector, 
			      Vector& d2vector);
    
    /// Set initial velocity 
    void SetInitialVelocity(Vector& xvel);

    
  private:
    
    /// Compute the marker vector phi
    void ComputePhi(Vector& phi);

    /// Compute the partial stress alpha
    void ComputeAlpha(Vector& alpha, Vector& phi, real& nu);
    
    /// Compute the density vector rho
    void ComputeDensity(Vector& rho, Vector& phi);

    /// Reassemble Dot Sigma
    void UpdateDotSigma(Vector& dsig, 
			BilinearForm& a, 
			LinearForm& L, 
			Matrix& A_tmp, 
			Vector& m_tmp);

    /// Returns true if vertex neighboring cells are all fluid
    bool isFluid(Vertex& vertex, const Vector& phi_vec) const;
   
    /// Returns true if vertex is not in boundary
    bool isInterior(Vertex& vertex) const;
   
    /// Returns vector component (i,j)
    unsigned int getij(unsigned int j, unsigned int i) const;

    /// Move the structure part of the mesh according to velocity u
    void UpdateStructure(Vector& mvel_vec, const Vector& phi_vec, Function& u);
    
    /// Smooth the fluid mesh
    void UpdateFluid(Vector& mvel_vec, const Vector& phi_vec);
    
    Mesh& mesh;
    Function& f;
    ALEBoundaryCondition& bc_mom;
    ALEBoundaryCondition& bc_con;
    Function& bisect;
    real& rhof;
    real& rhos;
    real& E;
    real& elnu;
    real& nu;
    real& T;
    real& k;

    MeshFunction<unsigned int> boundary_vertex_map;
    MeshFunction<unsigned int> boundary_cell_map;
    MeshFunction<bool>         vertex_is_interior;
    BoundaryMesh*              boundary;
   

  };
  //-----------------------------------------------------------------------------
  // INLINE: Function definitions - these functions are here because
  //                                they are used in every time step
  //-----------------------------------------------------------------------------
  inline bool FSISolver::isFluid(Vertex& vertex, const Vector& phi_vec) const
  {
    for (CellIterator cell(vertex); !cell.end(); ++cell) 
      if (phi_vec(cell->index()) == 0) return false;
    
    return true;
  }
  //-----------------------------------------------------------------------------
  inline bool FSISolver::isInterior(Vertex& vertex) const
  {  
    return (vertex_is_interior(vertex));
  }
  //-----------------------------------------------------------------------------
  inline unsigned int FSISolver::getij(unsigned int j, unsigned int i) const
  {
    return (i + mesh.topology().dim()*j);
  }
  //-----------------------------------------------------------------------------
  inline void FSISolver::UpdateStructure(Vector& mvel_vec, const Vector& phi_vec, Function& u) 
  {
    unsigned int ndx;

    for (VertexIterator v(mesh); !v.end(); ++v) {

      if (isFluid(*v, phi_vec)) continue;

      for (ndx = 0; ndx < mesh.topology().dim(); ndx++) {

	v->coordinates()[ndx] += k * u(*v, ndx);
	mvel_vec.set(getij(v->index(), ndx), mvel_vec.get(getij(v->index(), ndx)) + k * u(*v, ndx));
      }
    }
  }
  //-----------------------------------------------------------------------------
  inline void FSISolver::UpdateFluid(Vector& mvel_vec, const Vector& phi_vec)
  {
    real old_coord[3];
    unsigned int ndx;

    for (VertexIterator v(mesh); ! v.end(); ++v) {  
      
      if (!isFluid(*v, phi_vec)) continue;
      if (!isInterior(*v))       continue;
      
      for (ndx = 0; ndx < mesh.topology().dim(); ndx++) {
	
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
	mvel_vec.set(getij(v->index(), ndx), mvel_vec.get(getij(v->index(), ndx)) + (v->coordinates()[ndx] - old_coord[ndx]));	
      }	
    }
  }
  //-----------------------------------------------------------------------------
  inline void FSISolver::UpdateDotSigma(Vector& dsig, BilinearForm& a, LinearForm& L, Matrix& A_tmp, Vector& m_tmp)
  {
    
    FEM::assemble(a, A_tmp, mesh);
    FEM::assemble(L, dsig, mesh);
    FEM::lump(A_tmp, m_tmp);
    dsig.div(m_tmp); 
  }
  
}

#endif
