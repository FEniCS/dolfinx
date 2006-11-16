// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __PLASTICITY_PROBLEM_H
#define __PLASTICITY_PROBLEM_H

#include <dolfin/Solver.h>
#include <dolfin/NonlinearProblem.h>
#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/PlasticityModel.h>
#include <dolfin/ReturnMapping.h>

namespace dolfin
{

  class PlasticityProblem : public NonlinearProblem
  {
  public:

    /// Constructor
    PlasticityProblem(Function& u, Function& b, Mesh& mesh, 
          BoundaryCondition& bc, bool& elastic_tangent, PlasticityModel& plastic_model);

    /// Destructor 
    ~PlasticityProblem();

    /// Assemble Jacobian and residual vector 
    void form(GenericMatrix& A, GenericVector& b, const GenericVector& x);
    
    friend class PlasticitySolver;

  private:

    /// Class variables
    BilinearForm* a;
    BilinearForm* a_strain;
    BilinearForm* a_tangent;
    BilinearForm* a_plastic_strain;
    BilinearForm* a_equivalent_plastic_strain;
    LinearForm* L;
    LinearForm* L_strain;
    Function* plastic_strain_old_function;
    Function* plastic_strain_new_function;
    Function* equivalent_plastic_strain_old_function;
    Function* equivalent_plastic_strain_new_function;
    Function* consistent_tangent_old_function;
    Function* consistent_tangent_new_function;
    Function* strain_function;
    Function* stress_function;
    Mesh* _mesh;
    BoundaryCondition* _bc;
    bool* _elastic_tangent;
    PlasticityModel* _plastic_model;
    ReturnMapping* return_mapping;
    Matrix A_strain;
  };
}

#endif
