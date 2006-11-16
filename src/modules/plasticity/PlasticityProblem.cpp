// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin/PlasticityProblem.h>
#include <dolfin/Plas2D.h>
#include <dolfin/Strain2D.h>
#include <dolfin/Tangent2D.h>
#include <dolfin/p_strain2D.h>
#include <dolfin/ep_strain2D.h>
#include <dolfin/Plas3D.h>
#include <dolfin/Strain3D.h>
#include <dolfin/Tangent3D.h>
#include <dolfin/p_strain3D.h>
#include <dolfin/ep_strain3D.h>

using namespace dolfin;

PlasticityProblem::PlasticityProblem(Function& u, Function& b, Mesh& mesh, 
          BoundaryCondition& bc, bool& elastic_tangent, PlasticityModel& plastic_model):
          NonlinearProblem(), _mesh(&mesh), _bc(&bc), _elastic_tangent(&elastic_tangent), _plastic_model(&plastic_model)
{
  // Create functions
  plastic_strain_old_function = new Function;
  plastic_strain_new_function = new Function;
  equivalent_plastic_strain_old_function = new Function;
  equivalent_plastic_strain_new_function = new Function;
  consistent_tangent_old_function = new Function;
  consistent_tangent_new_function = new Function;
  strain_function = new Function;
  stress_function = new Function;

  // Create forms 2D or 3D
  if(mesh.topology().dim() == 2)
  {
    a = new Plas2D::BilinearForm(*consistent_tangent_new_function);
    a_strain = new Strain2D::BilinearForm;
    a_tangent = new Tangent2D::BilinearForm;
    a_plastic_strain = new p_strain2D::BilinearForm;
    a_equivalent_plastic_strain = new ep_strain2D::BilinearForm;
    L = new Plas2D::LinearForm(b, *stress_function);
    L_strain = new Strain2D::LinearForm(u);
  }
  else if(mesh.topology().dim() == 3)
  {
    a = new Plas3D::BilinearForm(*consistent_tangent_new_function);
    a_strain = new Strain3D::BilinearForm;
    a_tangent = new Tangent3D::BilinearForm;
    a_plastic_strain = new p_strain3D::BilinearForm;
    a_equivalent_plastic_strain = new ep_strain3D::BilinearForm;
    L = new Plas3D::LinearForm(b, *stress_function);
    L_strain = new Strain3D::LinearForm(u);
  }

  // Assemple matrix for strain computation, since it is constant it can be pre computed
  FEM::assemble(*a_strain, A_strain, mesh);

  // Initialise functions
  u.init(mesh, a->trial());
  plastic_strain_old_function->init(mesh, a_plastic_strain->trial());
  plastic_strain_new_function->init(mesh, a_plastic_strain->trial());
  consistent_tangent_old_function->init(mesh, a_tangent->trial());
  consistent_tangent_new_function->init(mesh, a_tangent->trial());
  equivalent_plastic_strain_old_function->init(mesh, a_equivalent_plastic_strain->test());
  equivalent_plastic_strain_new_function->init(mesh, a_equivalent_plastic_strain->test());
  strain_function->init(mesh, L_strain->test());
  stress_function->init(mesh, L_strain->test());

  return_mapping = new ReturnMapping;
}
//-----------------------------------------------------------------------------
PlasticityProblem::~PlasticityProblem()
{
  delete strain_function; delete stress_function;
  delete consistent_tangent_old_function;  delete consistent_tangent_new_function;
  delete equivalent_plastic_strain_old_function; delete equivalent_plastic_strain_new_function;
  delete plastic_strain_old_function; delete plastic_strain_new_function;
  delete a; delete a_strain; delete a_tangent; delete a_plastic_strain; delete a_equivalent_plastic_strain;
  delete L; delete L_strain; delete return_mapping;
}
//-----------------------------------------------------------------------------
void PlasticityProblem::form(GenericMatrix& A, GenericVector& b, const GenericVector& x)
{
  // Get vectors from functions
  Vector& plastic_strain_old = plastic_strain_old_function->vector();
  Vector& plastic_strain_new = plastic_strain_new_function->vector();
  Vector& equivalent_plastic_strain_old = equivalent_plastic_strain_old_function->vector();
  Vector& equivalent_plastic_strain_new = equivalent_plastic_strain_new_function->vector();
  Vector& consistent_tangent_old = consistent_tangent_old_function->vector();
  Vector& consistent_tangent_new = consistent_tangent_new_function->vector();
  Vector& strain = strain_function->vector();
  Vector& stress = stress_function->vector();

  // Compute strains
  LU solver;
  Vector b_strain;
  FEM::assemble(*L_strain, b_strain, *_mesh);
  solver.solve(A_strain, strain, b_strain);

  uint N(strain_function->vector().size()/strain_function->vectordim()), tangent_entry(0);
  real equivalent_plastic_strain(0.0);
  uBlasVector trial_stress(6), plastic_strain(6), elastic_strain(6), total_strain(6);
  uBlasDenseMatrix consistent_tangent(6,6);
  consistent_tangent.clear();
  plastic_strain.clear();
  elastic_strain.clear();
  total_strain.clear();

  for (uint m = 0; m != N; ++m)
  {
    // Elastic tangent is used in the first timestep
    if (*_elastic_tangent == true)
      consistent_tangent.assign(_plastic_model->elastic_tangent);

    // Consistent tangent from previous converged time step solution is used
    // as and initial guess.
    // 2D 
    if (*_elastic_tangent != true && _mesh->topology().dim() == 2)
    {
      consistent_tangent(0,0) = consistent_tangent_old(m);                
      consistent_tangent(0,1) = consistent_tangent_old(m + N);
      consistent_tangent(0,3) = consistent_tangent_old(m + 2*N);                
      consistent_tangent(1,0) = consistent_tangent_old(m + 3*N);
      consistent_tangent(1,1) = consistent_tangent_old(m + 4*N);                
      consistent_tangent(1,3) = consistent_tangent_old(m + 5*N);
      consistent_tangent(3,0) = consistent_tangent_old(m + 6*N);
      consistent_tangent(3,1) = consistent_tangent_old(m + 7*N);                
      consistent_tangent(3,3) = consistent_tangent_old(m + 8*N);
    }
    // 3D
    else if(*_elastic_tangent != true && _mesh->topology().dim() == 3)
    {
      tangent_entry = 0;
      for (int i = 0; i!=6; ++i)
        for (int j = 0; j!=6; ++j)
        {
          consistent_tangent(i,j) = consistent_tangent_old(m + tangent_entry*N);
          tangent_entry++;
        }
    }

    // Get plastic strain from previous converged time step
    for (int i = 0; i!=6; ++i)
      plastic_strain(i) = plastic_strain_old(m + i*N);

    // Get strains 2D or 3D
    if(_mesh->topology().dim() == 2)
    {
      total_strain(0) = strain(m);
      total_strain(1) = strain(m + N);
      total_strain(3) = strain(m + 2*N);
    }
    else if(_mesh->topology().dim() == 3)
      for (int i = 0; i!=6; ++i)
        total_strain(i) = strain(m + i*N);

    // Compute elastic strains        
    elastic_strain.assign(total_strain-plastic_strain);

    // Get equivalent plastic strain from previous converged time step
    equivalent_plastic_strain = equivalent_plastic_strain_old(m);
      
    // Trial stresses
    trial_stress.assign(prod(_plastic_model->elastic_tangent, elastic_strain));

    // Testing trial stresses, if yielding occurs the stresses are mapped 
    // back onto the yield surface, and the updated parameters are returned.
    return_mapping->ClosestPoint(*_plastic_model, consistent_tangent, trial_stress, plastic_strain, equivalent_plastic_strain);

    // Updating plastic strain 
    for (int i = 0; i!=6; ++i)
      plastic_strain_new(m + i*N) =  plastic_strain(i);

    // Updating equivalent plastic strain 
    equivalent_plastic_strain_new(m) = equivalent_plastic_strain;

    // Update stresses for next Newton iteration (trial stresses if elastic, otherwise current stress sig_c)
    // and coefficients for consistent tangent matrix 2D or 3D
    if(_mesh->topology().dim() == 2)
    {
      stress(m)        = trial_stress(0);
      stress(m + N)    = trial_stress(1);
      stress(m + 2*N)  = trial_stress(3);

      consistent_tangent_new(m)        = consistent_tangent(0,0);                
      consistent_tangent_new(m + N)    = consistent_tangent(0,1);
      consistent_tangent_new(m + 2*N)  = consistent_tangent(0,3);                
      consistent_tangent_new(m + 3*N)  = consistent_tangent(1,0);
      consistent_tangent_new(m + 4*N)  = consistent_tangent(1,1);                
      consistent_tangent_new(m + 5*N)  = consistent_tangent(1,3);
      consistent_tangent_new(m + 6*N)  = consistent_tangent(3,0);
      consistent_tangent_new(m + 7*N)  = consistent_tangent(3,1);                
      consistent_tangent_new(m + 8*N)  = consistent_tangent(3,3);
    }
    else if(_mesh->topology().dim() == 3)
    {
      for (int i = 0; i!=6; ++i)
        stress(m + i*N) = trial_stress(i); 

      tangent_entry = 0;
      for (int i = 0; i!=6; ++i)
        for (int j = 0; j!=6; ++j)
        {
          consistent_tangent_new(m + tangent_entry*N) = consistent_tangent(i,j);
          tangent_entry++;
        }
    }
  }

  // Assemble
  FEM::assemble(*a, *L, A, b, *_mesh);
  FEM::applyBC(A, *_mesh, a->test(), *_bc);
  FEM::applyResidualBC(b, x, *_mesh, a->test(), *_bc);

}
//-----------------------------------------------------------------------------    
