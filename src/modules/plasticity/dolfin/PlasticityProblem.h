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
#include "PlasticityModel.h"
#include <fstream>

namespace dolfin
{

  class PlasticityProblem : public NonlinearProblem
  {
  public:
    //constructor
    PlasticityProblem(Function& u, Function& b, Mesh& mesh, 
          BoundaryCondition& bc, bool& elastic_tangent, PlasticityModel& plas, 
          uBlasDenseMatrix& D);

    // Destructor 
    ~PlasticityProblem();

    // Assemble Jacobian and residual vector 
    void form(GenericMatrix& A, GenericVector& b, const GenericVector& x);
    
    Function* p_strain_old;
    Function* p_strain_new;

    Function* eq_strain_old;
    Function* eq_strain_new;

    Function* tangent_old;
    Function* tangent_new;

    Function* stress;

  private:

    // Pointers to forms, mesh data and boundary conditions
    BilinearForm* a;
    LinearForm* L;
    Matrix A_strain;
    LinearForm* L_strain;
    BilinearForm* a_strain;
    BilinearForm* a_tan;
    BilinearForm* ap_strain;
    BilinearForm* aep_strain;
    Mesh* _mesh;
    BoundaryCondition* _bc;
    Function* strain;
    bool* _elastic_tangent;
    PlasticityModel* _plas;
    uBlasDenseMatrix* _D;
  };

}  // end dolfin namespace

#endif
