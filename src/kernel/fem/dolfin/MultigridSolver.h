// Copyright (C) 2004 Andreas Mark and Andreas Nilsson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#ifndef __MULTIGRID_SOLVER_H
#define __MULTIGRID_SOLVER_H

namespace dolfin
{
  class MeshHierarchy;
  class Matrix;
  class Vector;
  
  /// The multigrid solver solves a given (Poisson-like) PDE using a hierarchy of
  /// discretizations. It has the advantage over other methods that it scales linearly
  /// with the number of unknowns.
  
  class MultigridSolver {
  public: 

    static void solve(PDE& pde, Vector& x, MeshHierarchy& meshes);    
    static void solve(PDE& pde, Vector& x, Mesh& mesh);
    static void solve(PDE& pde, Vector& x, Mesh& mesh, unsigned int refinements);
    
  private:
    
    typedef NewArray<Matrix> Matrices;
    typedef NewArray<Vector> Vectors;
    
    /// This is the main function, it assembles the matrices for the
    /// different meshes and then starts the main iteration using the
    /// initial guess x = 0. In the iteration the residual is
    /// calculated and then the fullVCycle is called. The iteration
    /// stops when either the given tolerance is reached or the
    /// maximum number of iterations has been exceeded.
    static void mainIteration(PDE& poisson, Vector& x, const MeshHierarchy& meshes);
    
    /// This function solves the initial coarse problem exactly and
    /// then iterpolates the residual to a finer mesh before it calls
    /// vCycle with an initial guess and new right hand side.
    static void fullVCycle(Vector& x, const Vector& r_fine, const MeshHierarchy& meshes, 
			   const Matrices& A);    
    
    /// This function starts with the problem on a fine grid and then
    /// improves the solution by damping the high-frequency error. Then
    /// the function calculates the residual which is restricted
    /// to the coarser mesh. In the next step the function solves the
    /// coarser problem recursively, with a zero initial guess and the
    /// residual as a new right hand side. Then the function
    /// interpolates the solution back to the finer grid. Finally the
    /// function improves the solution and adds the correction to the
    /// initial guess.
    static void vCycle(Vector& x, const Vector& rhs, const MeshHierarchy& meshes, 
		       const Matrices& A, unsigned int level);
    
    /// Improve solution by making a number of Gauss-Seidel iterations 
    /// which decreases the high-frequency error.
    static void smooth(Vector& x, const Vector& rhs, unsigned int noSmooths, 
		       const Matrices& A, unsigned int level);
    
    /// Restrict the residual to a coarser mesh.
    static void restrict(Vector& x, const MeshHierarchy& meshes, unsigned int level);

    /// Interpolate the residual to a finer mesh using linear interpolation.
    static void interpolate(Vector& x, const MeshHierarchy& meshes, unsigned int level);

  };
  
} 

#endif
