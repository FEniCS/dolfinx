// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SOLVER_NAVIER_STOKES_HH
#define __SOLVER_NAVIER_STOKES_HH

#include <kw_constants.h>
#include "Solver.hh"

namespace dolfin{

  class Grid;
  class SparseMatrix;
  class Vector;
  class OpenDX;
  class MatlabOld;
  class GlobalField;
  class Discretiser;
  class Equation;
  class Output;
  
  enum initialDataNS { zero_initial_data, white_noise, poiseuille_flow, couette_flow };
  
  ///
  class SolverNavierStokes: public Solver {
  public:
	 
	 /// Constructor
	 SolverNavierStokes(Grid *grid);
	 
	 /// Destructor
	 ~SolverNavierStokes();
	 
	 /// Solve transient problem
	 void solve();
	 
	 const char *Description();
	 
  private:
	 
	 /// Checks if the solution has reached a steady state
	 bool CheckIfStationary();
	 /// Check if velocity (U) - pressure (P) iteration has converged 
	 bool CheckUPConvergence(real pre_res_norm_2, real pre_res_norm_inf, 
									 real vel_res_norm_2, real vel_res_norm_inf );
	 /// Check if non linear iteration has converged 
	 bool CheckNonLinearConvergence(real pre_res_norm_2, real pre_res_norm_inf, 
											  real vel_res_norm_2, real vel_res_norm_inf );
	 
	 /// Set initial data
	 void SetInitialData();
	 
	 /// Set tolerances for solution iterations
	 void SetTolerances();

	 void WriteDataToFile();
	 void WriteSolutionToFile();
	 /// Write pertubation data to file
	 void WritePertubationData();
	 /// Write solution data to file
	 void WriteData();
	 // Compute functionals 
	 void ComputeFunctionals();
	 /// Compute continuous residual 
	 void CompContResidual();
	 void WriteReynoldsStresses();
	 
	 /// Non linear problem using fixed point iteration
	 void SolveNonLinearProblemUsingFixPoint(Discretiser *discrete_vel, Discretiser *discrete_pre, Equation *eq_vel, Equation *eq_pre);
	 /// Solve non linear problem using Newton-GMRES
	 void SolveNonLinearProblemUsingGMRES(Discretiser *discrete_vel, Discretiser *discrete_pre, Equation *eq_vel, Equation *eq_pre);
	 
	 /// Set parameters for subgrid modeling  
	 void SetSubgridModelingUtilities();
	 
	 void ReAssembleVelMatrix(bool am);
	 
	 //void InitEquationData(Equation *equation);
	 
	 void TimestepVelocity(Discretiser *discrete, Equation *equation);
	 void TimestepPressure(Discretiser *discrete, Equation *equation);
	 
	 void ComputeDiscreteResidual(Discretiser *discrete, Equation *equation, 
											SparseMatrix *A, Vector *U, real &norm_2, real &norm_inf);
	 
	 void Reset();
	 
	 bool SaveSample();
	 int  current_frame;
	 
	 void InitSolutionVariables();
	 void DeleteSolutionVariables();
	 
	 real GetTimeStep();
	 void GetMeshSize();
	 real LocalMeshSize(int el);
	 
	 void SetTurbulentInflow(real trb);
	 
	 // Inner variables
	 OpenDX *opendx;
	 OpenDX *opendx_residuals;
	 
	 Output *output;
	 
	 initialDataNS initial_data;
	 
	 real progress;
	 
	 GlobalField *upTS_field;
	 GlobalField *ppTS_field;
	 GlobalField *upNL_field;
	 GlobalField *ppNL_field;
	 GlobalField *u_field;
	 GlobalField *p_field;
	 
	 GlobalField *dt_field;
	 GlobalField *t_field;
	 GlobalField *Re_field;
	 
	 GlobalField *tau_field;
	 
	 GlobalField *uc_field;
	 GlobalField *pc_field;
	 GlobalField *uf_field;
	 GlobalField *pf_field;
	 
	 GlobalField *up_field;
	 
	 Vector *u;
	 Vector *p;
	 Vector *upTS;
	 Vector *ppTS;
	 Vector *upNL;
	 Vector *ppNL;
	 Vector *print_vector;
	 Vector *u_fine;
	 Vector *p_fine;
	 Vector *u_coarse;
	 Vector *p_coarse;
	 Vector *tau;
	 Vector *tau_print;
	 
	 SparseMatrix *A_pre;
	 SparseMatrix *A_vel;
	 
	 bool A_vel_assembled;
	 bool A_pre_assembled;
	 bool re_assemble_matrix;
	 
	 int N;
	 
	 MatlabOld *matlab_data;
	 
	 int no_data_entries, no_residual_entries, no_pertubation_entries, no_re_stress_entries;
	 
	 int write_data;
	 int write_residuals;
	 int compute_projections;
	 int compute_reynolds_stresses;
	 int write_reynolds_stresses;  
	 int turbulent_flow;
	 int pertubation,pert_poiseuille,pert_couette;
	 int subgrid_model_on_divergence_form;
	 
	 int nsd,no_nodes,no_cells,print_no_nodes,print_no_cells;
	 int noeq,noeq_vel,noeq_pre;
	 real grid_h,t,dt,T,T0,Re;
	 
	 bool UP_conv,NL_conv;
	 int NL_it,UP_it,max_no_UP_iter,max_no_NL_iter;
	 real NL_tol,UP_tol,NL_max_tol,UP_max_tol,stat_tol;
	 real UP_res_norm_2, UP_res_norm_inf, NL_res_norm_2, NL_res_norm_inf;
	 real maxnorm_du;
  };

}
  
#endif
