// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include "dolfin/PlasticityProblem.h"

#include "dolfin/Plas2D.h"
#include "dolfin/Strain2D.h"
#include "dolfin/Tangent2D.h"
#include "dolfin/p_strain2D.h"
#include "dolfin/ep_strain2D.h"
#include "dolfin/Plas3D.h"
#include "dolfin/Strain3D.h"
#include "dolfin/Tangent3D.h"
#include "dolfin/p_strain3D.h"
#include "dolfin/ep_strain3D.h"

using namespace dolfin;

PlasticityProblem::PlasticityProblem(Function& u, Function& b, Mesh& mesh, 
          BoundaryCondition& bc, bool& elastic_tangent, PlasticityModel& plas, 
          uBlasDenseMatrix& D) : NonlinearProblem(),
          _mesh(&mesh), _bc(&bc), _elastic_tangent(&elastic_tangent), _plas(&plas), _D(&D)
{
  // Create functions
  strain = new Function; stress = new Function;
  tangent_old = new Function;  tangent_new = new Function;
  eq_strain_old = new Function; eq_strain_new = new Function;
  p_strain_old = new Function; p_strain_new = new Function;

  // Create forms 2D or 3D
  if(mesh.topology().dim() == 2)
  {
    a = new Plas2D::BilinearForm(*tangent_new);
    L = new Plas2D::LinearForm(b, *stress);
    a_strain = new Strain2D::BilinearForm;
    L_strain = new Strain2D::LinearForm(u);
    a_tan = new Tangent2D::BilinearForm;
    ap_strain = new p_strain2D::BilinearForm;
    aep_strain = new ep_strain2D::BilinearForm;
  }
  else if(mesh.topology().dim() == 3)
  {
    a = new Plas3D::BilinearForm(*tangent_new);
    L = new Plas3D::LinearForm(b, *stress);
    a_strain = new Strain3D::BilinearForm;
    L_strain = new Strain3D::LinearForm(u);
    a_tan = new Tangent3D::BilinearForm;
    ap_strain = new p_strain3D::BilinearForm;
    aep_strain = new ep_strain3D::BilinearForm;
  }

  // assemple matrix for strain computation, since it is constant it can be pre computed
  FEM::assemble(*a_strain, A_strain, mesh);

  // initialise functions
  u.init(mesh, a->trial());
  strain->init(mesh, L_strain->test());
  stress->init(mesh, L_strain->test());
  tangent_old->init(mesh, a_tan->trial());
  tangent_new->init(mesh, a_tan->trial());
  eq_strain_old->init(mesh, aep_strain->test());
  eq_strain_new->init(mesh, aep_strain->test());
  p_strain_old->init(mesh, ap_strain->trial());
  p_strain_new->init(mesh, ap_strain->trial());
}
//-----------------------------------------------------------------------------
PlasticityProblem::~PlasticityProblem()
{
  delete strain; delete stress;
  delete tangent_old;  delete tangent_new;
  delete eq_strain_old; delete eq_strain_new;
  delete p_strain_old; delete p_strain_new;
  delete a; delete L;
  delete a_strain; delete L_strain; delete a_tan;
}
//-----------------------------------------------------------------------------
void PlasticityProblem::form(GenericMatrix& A, GenericVector& b, const GenericVector& x)
{
  // Get vectors from functions
  Vector& eps = strain->vector();
  Vector& sig = stress->vector();
  Vector& eq_eps_old = eq_strain_old->vector();
  Vector& eq_eps_new = eq_strain_new->vector();
  Vector& Tan_old = tangent_old->vector();
  Vector& Tan_new = tangent_new->vector();
  Vector& eps_p_old = p_strain_old->vector();
  Vector& eps_p_new = p_strain_new->vector();

  // compute strains
  LU solver;
  Vector b_strain;
  FEM::assemble(*L_strain, b_strain, *_mesh);
  solver.solve(A_strain, eps, b_strain);

  int N(strain->vector().size()/strain->vectordim()), n(6), ntan(0);
  real eps_eq(0);
  uBlasVector t_sig(6), eps_p(6), eps_e(6), eps_t(6);
  uBlasDenseMatrix cons_t(6,6);
  cons_t.clear();
  eps_p.clear();
  eps_e.clear();
  eps_t.clear();

  for (int m = 0; m != N; ++m)
  {
    // elastic tangent is used in the first timestep
    if (*_elastic_tangent == true)
      cons_t.assign(*_D);

    // consistent tangent from previous converged time step solution is used
    // as and initial guess.
    if (*_elastic_tangent != true )
    {
      // 2D 
      if(_mesh->topology().dim() == 2)
      {
        cons_t(0,0) = Tan_old(m);                
        cons_t(0,1) = Tan_old(m + N);
        cons_t(0,3) = Tan_old(m + 2*N);                
        cons_t(1,0) = Tan_old(m + 3*N);
        cons_t(1,1) = Tan_old(m + 4*N);                
        cons_t(1,3) = Tan_old(m + 5*N);
        cons_t(3,0) = Tan_old(m + 6*N);
        cons_t(3,1) = Tan_old(m + 7*N);                
        cons_t(3,3) = Tan_old(m + 8*N);
      }
      // 3D
      else if(_mesh->topology().dim() == 3)
      {
        ntan = 0;
        for (int i = 0; i!=n; ++i)
        {
          for (int j = 0; j!=n; ++j)
          {
            cons_t(i,j) = Tan_old(m + ntan*N);
            ntan++;
          }
        }
      }
    }

    // get plastic strain from previous converged time step
    for (int i = 0; i!=n; ++i)
      eps_p(i) = eps_p_old(m + i*N);

    // get strains 2D or 3D
    if(_mesh->topology().dim() == 2)
    {
      eps_t(0) = eps(m);
      eps_t(1) = eps(m + N);
      eps_t(3) = eps(m + 2*N);
    }
    else if(_mesh->topology().dim() == 3)
    {
      for (int i = 0; i!=n; ++i)
        eps_t(i) = eps(m + i*N);
    }

    // compute elastic strains        
    eps_e.assign(eps_t-eps_p);

    // get equivalent plastic strain from previous converged time step
    eps_eq = eq_eps_old(m);
      
    // trial stresses
    t_sig.assign(prod(*_D, eps_e));

    // testing trial stresses, if yielding occurs the stresses are mapped 
    // back onto the yield surface, and the updated parameters are returned.
    _plas->return_mapping(cons_t, *_D, t_sig, eps_p, eps_eq);

    // updating plastic strain 
    for (int i = 0; i!=n; ++i)
      eps_p_new(m + i*N) =  eps_p(i);

    // updating equivalent plastic strain 
    eq_eps_new(m) = eps_eq;

    // update stresses for next Newton iteration (trial stresses if elastic, otherwise current stress sig_c)
    // and coefficients for consistent tangent matrix 2D or 3D
    if(_mesh->topology().dim() == 2)
    {
      sig(m)        =  t_sig(0);
      sig(m + N)    = t_sig(1);
      sig(m + 2*N)  = t_sig(3);

      Tan_new(m)        = cons_t(0,0);                
      Tan_new(m + N)    = cons_t(0,1);
      Tan_new(m + 2*N)  = cons_t(0,3);                
      Tan_new(m + 3*N)  = cons_t(1,0);
      Tan_new(m + 4*N)  = cons_t(1,1);                
      Tan_new(m + 5*N)  = cons_t(1,3);
      Tan_new(m + 6*N)  = cons_t(3,0);
      Tan_new(m + 7*N)  = cons_t(3,1);                
      Tan_new(m + 8*N)  = cons_t(3,3);
    }
    else if(_mesh->topology().dim() == 3)
    {
      for (int i = 0; i!=n; ++i)
        sig(m + i*N) = t_sig(i); 

      ntan = 0;
      for (int i = 0; i!=n; ++i)
      {
        for (int j = 0; j!=n; ++j)
        {
          Tan_new(m + ntan*N) = cons_t(i,j);
          ntan++;
        }
      }
    }

  }

  // assemble
  FEM::assemble(*a, *L, A, b, *_mesh);
  FEM::applyBC(A, *_mesh, a->test(), *_bc);
  FEM::applyResidualBC(b, x, *_mesh, a->test(), *_bc);

}
//-----------------------------------------------------------------------------    
