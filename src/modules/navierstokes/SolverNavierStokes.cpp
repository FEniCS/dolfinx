// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "ProblemNS.h"
#include "EquationVel_cG1cG1.h"
#include "EquationPre_cG1cG1.h"
#include "EquationVel2d_cG1cG1.h"
#include "EquationPre2d_cG1cG1.h"
#include <Grid.h>
#include <Vector.h>
#include <SparseMatrix.h>
#include <MatlabOld.h>
#include <unistd.h>

//----------------------------------------------------------------------------------------------------
ProblemNS::ProblemNS(Grid *grid) : Problem(grid) 
{
  // Get space dimension
  settings->Get("space dimension",&nsd);
  if ( (nsd!=2) && (nsd!=3) )
    display->InternalError("ProblemNS::ProblemNS()","Only implemented for 2d and 3d");

  // Get inner variables
  settings->Get("write residuals", &write_residuals);
  settings->Get("write data", &write_data);
  settings->Get("compute projections",&compute_projections);
  settings->Get("write couette pertubations",&pert_couette);
  settings->Get("write poiseuille pertubations",&pert_poiseuille);
  pertubation = false;
  if ( pert_poiseuille || pert_couette ) pertubation = true;

  // Initialize Opendx objects
  char output_file[DOLFIN_LINELENGTH];
  settings->Get("output file primal",output_file);

  //opendx_residuals = new OpenDX(output_file,4,nsd,1,1,1);
  //opendx_residuals->SetLabel(0,"R1");
  //opendx_residuals->SetLabel(1,"R2");
  //opendx_residuals->SetLabel(1,"R3");
  //opendx_residuals->SetLabel(1,"R4");

  // Initialise Output for 2d problems
  if ( nsd == 2 ){
    output = new Output("navier_stokes_2d.m",2,nsd,1);
  }
  
  // Initialize scalar output data 
  if ( nsd!=3 ){
    write_data = false;
    write_residuals = false;
    pertubation = false;
    write_reynolds_stresses = false;
  }

  no_data_entries = 0;
  no_residual_entries = 0;
  no_pertubation_entries = 0;
  no_re_stress_entries = 0;
  if ( write_data )               no_data_entries        = 39;
  if ( write_residuals)           no_residual_entries    = 12;
  if ( pertubation )              no_pertubation_entries = 24;
  if ( write_reynolds_stresses )  no_re_stress_entries   = 27;
  matlab_data = NULL;
  if ( write_data || write_residuals || pertubation || write_reynolds_stresses ){
    matlab_data = new MatlabOld(no_data_entries + no_residual_entries + no_pertubation_entries + no_re_stress_entries);
  }
  
  if ( write_data ){
    matlab_data->SetLabel(0,"|| u ||_1");
    matlab_data->SetLabel(1,"|| u ||_2");
    matlab_data->SetLabel(2,"|| u ||_{max}");
    matlab_data->SetLabel(3,"| u |_{1,1}");
    matlab_data->SetLabel(4,"| u |_{1,2}");
    matlab_data->SetLabel(5,"| u |_{1,max}");
    matlab_data->SetLabel(6,"|| u ||_{1,1}");
    matlab_data->SetLabel(7,"|| u ||_{1,2}");
    matlab_data->SetLabel(8,"|| u ||_{1,max}");
    matlab_data->SetLabel(9,"| u |_{2,1}");
    matlab_data->SetLabel(10,"| u |_{2,2}");
    matlab_data->SetLabel(11,"| u |_{2,max}");
    matlab_data->SetLabel(12,"|| u ||_{2,1}");
    matlab_data->SetLabel(13,"|| u ||_{2,2}");
    matlab_data->SetLabel(14,"|| u ||_{2,max}");
    matlab_data->SetLabel(15,"|| p ||_1");
    matlab_data->SetLabel(16,"|| p ||_2");
    matlab_data->SetLabel(17,"|| p ||_{max}");
    matlab_data->SetLabel(18,"| p |_{1,1}");
    matlab_data->SetLabel(19,"| p |_{1,2}");
    matlab_data->SetLabel(20,"| p |_{1,max}");
    matlab_data->SetLabel(21,"|| p ||_{1,1}");
    matlab_data->SetLabel(22,"|| p ||_{1,2}");
    matlab_data->SetLabel(23,"|| p ||_{1,max}");
    matlab_data->SetLabel(24,"| p |_{2,1}");
    matlab_data->SetLabel(25,"| p |_{2,2}");
    matlab_data->SetLabel(26,"| p |_{2,max}");
    matlab_data->SetLabel(27,"|| p ||_{2,1}");
    matlab_data->SetLabel(28,"|| p ||_{2,2}");
    matlab_data->SetLabel(29,"|| p ||_{2,max}");
    matlab_data->SetLabel(30,"|| div u ||_1");
    matlab_data->SetLabel(31,"|| div u ||_2");
    matlab_data->SetLabel(32,"|| div u ||_{max}");
    matlab_data->SetLabel(33,"|| du/dt ||_1");
    matlab_data->SetLabel(34,"|| du/dt ||_2");
    matlab_data->SetLabel(35,"|| du/dt ||_{max}");
    matlab_data->SetLabel(36,"|| dp/dt ||_1");
    matlab_data->SetLabel(37,"|| dp/dt ||_2");
    matlab_data->SetLabel(38,"|| dp/dt ||_{max}");
  }

  if ( write_residuals ){
    matlab_data->SetLabel(no_data_entries+0,"|| R1 ||_1");
    matlab_data->SetLabel(no_data_entries+1,"|| R1 ||_2");
    matlab_data->SetLabel(no_data_entries+2,"|| R1 ||_{max}");
    matlab_data->SetLabel(no_data_entries+3,"|| R2 ||_1");
    matlab_data->SetLabel(no_data_entries+4,"|| R2 ||_2");
    matlab_data->SetLabel(no_data_entries+5,"|| R2 ||_{max}");
    matlab_data->SetLabel(no_data_entries+6,"|| R3 ||_1");
    matlab_data->SetLabel(no_data_entries+7,"|| R3 ||_2");
    matlab_data->SetLabel(no_data_entries+8,"|| R3 ||_{max}");
    matlab_data->SetLabel(no_data_entries+9,"|| R4 ||_1");
    matlab_data->SetLabel(no_data_entries+10,"|| R4 ||_2");
    matlab_data->SetLabel(no_data_entries+11,"|| R4 ||_{max}");
  }

  if ( pertubation ){
    matlab_data->SetLabel(no_data_entries+no_residual_entries+0,"|| u1_{pert} ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+1,"|| u1_{pert} ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+2,"|| u2_{pert} ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+3,"|| u2_{pert} ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+4,"|| u3_{pert} ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+5,"|| u3_{pert} ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+6,"|| du_1/dx_1 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+7,"|| du_1/dx_1 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+8,"|| du_1/dx_2 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+9,"|| du_1/dx_2 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+10,"|| du_1/dx_3 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+11,"|| du_1/dx_3 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+12,"|| du_2/dx_1 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+13,"|| du_2/dx_1 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+14,"|| du_2/dx_2 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+15,"|| du_2/dx_2 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+16,"|| du_2/dx_3 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+17,"|| du_2/dx_3 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+18,"|| du_3/dx_1 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+19,"|| du_3/dx_1 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+20,"|| du_3/dx_2 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+21,"|| du_3/dx_2 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+22,"|| du_3/dx_3 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+23,"|| du_3/dx_3 ||_{max}");
  }

  if ( write_reynolds_stresses ){
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+0,"|| tau_11 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+1,"|| tau_12 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+2,"|| tau_13 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+3,"|| tau_22 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+4,"|| tau_23 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+5,"|| tau_33 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+6,"|| tau_11 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+7,"|| tau_12 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+8,"|| tau_13 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+9,"|| tau_22 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+10,"|| tau_23 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+11,"|| tau_33 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+12,"|| tau_11 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+13,"|| tau_12 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+14,"|| tau_13 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+15,"|| tau_22 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+16,"|| tau_23 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+17,"|| tau_33 ||_{max}");

    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+18,"|| div tau_1 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+19,"|| div tau_2 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+20,"|| div tau_3 ||_1");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+21,"|| div tau_1 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+22,"|| div tau_2 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+23,"|| div tau_3 ||_2");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+24,"|| div tau_1 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+25,"|| div tau_2 ||_{max}");
    matlab_data->SetLabel(no_data_entries+no_residual_entries+no_pertubation_entries+26,"|| div tau_3 ||_{max}");
  }
}
//----------------------------------------------------------------------------------------------------
ProblemNS::~ProblemNS()
{
}      
//-----------------------------------------------------------------------------
const char *ProblemNS::Description()
{
  return "Navier-Stokes equations";
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::solve()
{
  display->Status(0,"Solving Navier-Stokes primal problem");
  
  // Set Reynolds number 
  settings->Get("reynolds number",&Re);

  // Reset variables
  Reset();

  // Set FEM discretisation 
  Equation *eq_vel;
  Equation *eq_pre;

  if ( nsd == 3 ){
    eq_vel = new EquationVel_cG1cG1;
    eq_pre = new EquationPre_cG1cG1;
  }
  else{
    eq_vel = new EquationVel2d_cG1cG1;
    eq_pre = new EquationPre2d_cG1cG1;
  }
  
  Discretiser         discrete_vel(grid,eq_vel);
  Discretiser         discrete_pre(grid,eq_pre);

  // Set number of equations 
  noeq_vel = eq_vel->GetNoEq();
  noeq_pre = eq_pre->GetNoEq();
  noeq     = noeq_vel + noeq_pre;
 
  // Initialize solution variables
  InitSolutionVariables();  
  
  GlobalField upTS_field(grid,upTS,noeq_vel);
  GlobalField ppTS_field(grid,ppTS,noeq_pre);
  GlobalField upNL_field(grid,upNL,noeq_vel);
  GlobalField ppNL_field(grid,ppNL,noeq_pre);
  GlobalField u_field(grid,u,noeq_vel);
  GlobalField p_field(grid,p,noeq_pre);

  GlobalField uc_field(grid,u_coarse,noeq_vel);
  GlobalField pc_field(grid,u_coarse,noeq_pre);
  GlobalField uf_field(grid,u_fine,noeq_vel);
  GlobalField pf_field(grid,u_fine,noeq_pre);

  GlobalField *fx;
  GlobalField *fy;
  GlobalField *fz;

  // A global field for saving the solution
  up_field = new GlobalField(&u_field,&p_field);
  if ( nsd == 3 )
	 up_field->SetSize(2,3,1);
  else
	 up_field->SetSize(2,2,1);
  up_field->SetLabel("u","Velocity",0);
  up_field->SetLabel("p","Pressure",1);

  // A global field for the right-hand side
  fx = new GlobalField(grid,"fx"); 
  fy = new GlobalField(grid,"fy");
  if (nsd==3) fz = new GlobalField(grid,"fz");
  
  GlobalField tau_field(grid,tau,6);

  int field_no = 0;
  for (int i=0;i<noeq_vel;i++) eq_vel->AttachField(field_no++,&upTS_field,i);
  eq_vel->AttachField(field_no++,&ppTS_field);
  for (int i=0;i<noeq_vel;i++) eq_vel->AttachField(field_no++,&upNL_field,i);
  eq_vel->AttachField(field_no++,&ppNL_field);
  for (int i=0;i<noeq_vel;i++) eq_vel->AttachField(field_no++,&u_field,i);
  eq_vel->AttachField(field_no++,&p_field);
  for (int i=0;i<noeq_vel;i++) eq_vel->AttachField(field_no++,&uf_field,i);
  eq_vel->AttachField(field_no++,&pf_field);
  for (int i=0;i<noeq_vel;i++) eq_vel->AttachField(field_no++,&uc_field,i);
  eq_vel->AttachField(field_no++,&pc_field);
  eq_vel->AttachField(field_no++,fx);
  eq_vel->AttachField(field_no++,fy);
  if (nsd==3) eq_vel->AttachField(field_no++,fz);
  
  field_no = 0;
  for (int i=0;i<noeq_vel;i++) eq_pre->AttachField(field_no++,&upTS_field,i);
  eq_pre->AttachField(field_no++,&ppTS_field);
  for (int i=0;i<noeq_vel;i++) eq_pre->AttachField(field_no++,&upNL_field,i);
  eq_pre->AttachField(field_no++,&ppNL_field);
  for (int i=0;i<noeq_vel;i++) eq_pre->AttachField(field_no++,&u_field,i);
  eq_pre->AttachField(field_no++,&p_field);
  for (int i=0;i<noeq_vel;i++) eq_pre->AttachField(field_no++,&uf_field,i);
  eq_pre->AttachField(field_no++,&pf_field);
  for (int i=0;i<noeq_vel;i++) eq_pre->AttachField(field_no++,&uc_field,i);
  eq_pre->AttachField(field_no++,&pc_field);
  eq_pre->AttachField(field_no++,fx);
  eq_pre->AttachField(field_no++,fy);
  if (nsd==3) eq_pre->AttachField(field_no++,fz);
  
  // Reassemble the momentum matrix or not
  ReAssembleVelMatrix(true);
  
  // Set initial data
  //initial_data = zero_initial_data;
  initial_data = couette_flow;
  SetInitialData();  

  // Set time t to starting time T0
  t = T0;

  // Write data to file
  WriteDataToFile();  
  
  // Start time stepping
  display->Status(0,"Start computation at t = %.1f",t);
  while ( t <= T ){
    
    // Set time step
    GetMeshSize();
    dt = GetTimeStep();
    t += dt;

    // Set time and time step for equations
    eq_vel->SetTime(t);
    eq_pre->SetTime(t);
    eq_vel->SetTimeStep(dt);
    eq_pre->SetTimeStep(dt);
    
    // Save solution if we should
    if ( SaveSample() ) WriteSolutionToFile();
    
    // Set turbulent inflow condition
    SetTurbulentInflow(0.04);
    
    // Write some output for gdisp
    progress = (t-T0) / (T-T0);
    display->Progress(0,progress,"Primal solution: %.1f %%. Currently at t = %.4f, dt = %.4f",progress*100.0,t,dt);
    
    // Set solution from previous time step
    upTS->CopyFrom(u);
    // Set solution from previous outer non linear iteration
    upNL->CopyFrom(u);
    
    // Solve non linear problem
    SolveNonLinearProblemUsingFixPoint(&discrete_vel,&discrete_pre,eq_vel,eq_pre);
    //SolveNonLinearProblemUsingGMRES();
    
    // Write data to file
    WriteDataToFile();
    
    // Check if the solution is stationary
    if ( CheckIfStationary() ) break;
  }
      
  // Solution computed to end time
  display->Status(0,"Solution has been computed to endtime T = %f",t);
                               
  // Delete solution vectors
  DeleteSolutionVariables();

  delete eq_vel;
  delete eq_pre;

  delete fx;
  delete fy;
  delete fz;
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::SolveNonLinearProblemUsingFixPoint(Discretiser *discrete_vel, Discretiser *discrete_pre, 
						   Equation *eq_vel, Equation *eq_pre)
{
  real pre_res_norm_2, pre_res_norm_inf, vel_res_norm_2, vel_res_norm_inf;  
  
  // Start non linear iteration
  for (NL_it=0; NL_it < max_no_NL_iter; NL_it++){ 
      
    // Display information
    display->Status(0,"Starting non-linear iteration %d.",NL_it+1);
    
    // Compute Reynolds stresses - for subgrid model
    if ( compute_reynolds_stresses ){
      display->Status(5,"Estimate Reynolds stresses (Update subgrid model)");
      //scale_expol.EstimateReynoldsStresses( u, tau, &(*GRIDREL)(0), &(*GRIDREL)(1), 
      //				   &(*GRIDREL)(2), &(*GRIDREL)(3), &(*GRIDREL)(4) );
    }
    
    // Compute projections - for subgrid model
    if ( compute_projections ){
      display->Status(5,"Extract coarse and fine scale field");
      //(*GRIDREL)(0).projNodalData(u,u_coarse);
      //(*GRIDREL)(0).projNodalData(u_coarse);
      //for (int i=0; i < no_nodes*noeq_vel; i++ ) u_fine(i) = u(i) - u_coarse(i);
    }
    
    // Start inner iteration
    for (UP_it=0; UP_it < max_no_UP_iter; UP_it++){

		// Display information
		display->Status(1,"Starting u-p iteration %d.",UP_it+1);
		
      // Solve continuity equation
      display->Status(2,"Solving for the pressure.");
      TimestepPressure(discrete_pre,eq_pre);
      
      // Solve momentum equation
      display->Status(2,"Solving for the velocity.");
      TimestepVelocity(discrete_vel,eq_vel);
      
      // Re assemble the momentum matrix or not
      ReAssembleVelMatrix(false);
      
      // Compute residuals for inner iteration
      ComputeDiscreteResidual(discrete_pre,eq_pre,A_pre,p,pre_res_norm_2, pre_res_norm_inf);
      ComputeDiscreteResidual(discrete_vel,eq_vel,A_vel,u,vel_res_norm_2, vel_res_norm_inf);
      
      // Check if inner loop has converged
      if ( CheckUPConvergence(pre_res_norm_2, pre_res_norm_inf, vel_res_norm_2, vel_res_norm_inf) ) break;
    }
    
    // Re assemble the momentum matrix or not
    ReAssembleVelMatrix(true);
    
    // Compute residuals for non linear outer iteration 
    upNL->CopyFrom(u);
    ComputeDiscreteResidual(discrete_pre,eq_pre,A_pre,p,pre_res_norm_2, pre_res_norm_inf);
    ComputeDiscreteResidual(discrete_vel,eq_vel,A_vel,u,vel_res_norm_2,vel_res_norm_inf);
    
    // Re assemble the momentum matrix or not
    ReAssembleVelMatrix(false);
    
    // Check if outer loop has converged
    if ( CheckNonLinearConvergence(pre_res_norm_2, pre_res_norm_inf, vel_res_norm_2, vel_res_norm_inf) ) break;
  }
}
//----------------------------------------------------------------------------------------------------
void SolveNonLinearProblemUsingGMRES(Discretiser *discrete_vel, Discretiser *discrete_pre, 
				     Equation *eq_vel, Equation *eq_pre)
{
  display->InternalError("ProblemNS::SolveNonLinearProblemUsingGMRES()","Not implemented yet");
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::TimestepVelocity(Discretiser *discrete, Equation *equation)
{
  Vector b(no_nodes*equation->GetNoEq());
  
  // assemble matrix if not assembled, or reassemble if we should
  if ( !A_vel_assembled ){
    display->Status(3,"Assemble matrix A");
    discrete->AssembleLHS(A_vel);
    A_vel_assembled = true;
  } else if ( A_vel_assembled && re_assemble_matrix ){
    display->Status(3,"Re-assembling matrix A");
    discrete->AssembleLHS(A_vel);
  }

  // assemble vector
  display->Status(3,"Assemble vector b");
  discrete->AssembleRHS(&b);
  
  // set bc
  display->Message(2,"Load norm: %f",b.Norm(2));
  discrete->SetBoundaryConditions(A_vel,&b);
  display->Message(2,"Load norm: %f (with b.c.)",b.Norm(2));
  
  // solve discrete system  
  display->Status(3,"Solve algebraic system of equations");

  KrylovSolver solver;
  solver.SetMethod(gmres);
  solver.Solve(A_vel,u,&b);
  
  // compute norm of solution
  display->Message(4,"Norm of u = %f",u->Norm(2));
  display->Message(4,"Norm of A = %f",A_vel->Norm());
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::TimestepPressure(Discretiser *discrete, Equation *equation)
{
  Vector b(no_nodes*equation->GetNoEq());
  
  if ( !A_pre_assembled ){
    display->Status(3,"Assemble matrix A and vector b.");
    discrete->AssembleLHS(A_pre);
    A_pre_assembled = true;
  } 
  
  display->Status(3,"Assemble vector b");
  discrete->AssembleRHS(&b);

  // set dirichlet bc
  display->Message(2,"Load norm: %f",b.Norm(2));
  discrete->SetBoundaryConditions(A_pre,&b);
  display->Message(2,"Load norm: %f (with b.c.)",b.Norm(2));

  display->Status(3,"Solve algebraic system of equations");

  KrylovSolver solver;
  solver.SetMethod(gmres);
  solver.Solve(A_pre,p,&b);

  // compute norm of solution
  display->Message(4,"Norm of p = %f",p->Norm(2));
  display->Message(4,"Norm of A = %f",A_pre->Norm());
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::ComputeDiscreteResidual(Discretiser *discrete, Equation *equation, 
					SparseMatrix *A, Vector *U, real& norm_2, real& norm_inf)
{      
  Vector b(no_nodes*equation->GetNoEq());

  // Only for momentum equation
  if ( re_assemble_matrix && (equation->GetNoEq() == noeq_vel) ){
    display->Status(3,"Assemble matrix A");
    discrete->AssembleLHS(A);
  }

  display->Status(3,"Assemble vector b");
  discrete->AssembleRHS(&b);
    
  // set dirichlet bc
  display->Message(2,"Load norm: %f",b.Norm(2));
  discrete->SetBoundaryConditions(A,&b);
  display->Message(2,"Load norm: %f (with b.c.)",b.Norm(2));

  display->Message(4,"Norm of solution = %f",U->Norm(2));
  display->Message(4,"Norm of matrix  = %f",A->Norm());

  display->Message(3,"Compute discrete residual");
  Vector AU(no_nodes*equation->GetNoEq());
  A->Mult(U,&AU);
  
  Vector res(no_nodes*equation->GetNoEq());
  for (int i=0; i < no_nodes*equation->GetNoEq(); i++) res.Set(i,b(i) - AU(i));

  norm_2   = res.Norm(2);
  norm_inf = res.Norm(0);
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::WriteDataToFile()
{
  display->Status(0,"Write data to file");
  if ( write_data )      WriteData();
  if ( write_residuals ) CompContResidual();
  if ( pertubation )     WritePertubationData();
  if ( write_reynolds_stresses ) WriteReynoldsStresses();
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::WriteSolutionToFile()
{
  display->Status(0,"Write solution to file");

  up_field->Save(t);
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::ReAssembleVelMatrix(bool re_assemble_matrix)
{
  this->re_assemble_matrix = re_assemble_matrix;
}
//-----------------------------------------------------------------------------
bool ProblemNS::SaveSample()
{
  // Check if we should save a sample
  bool save_sample = false;
  
  int ops;
  settings->Get("output samples", &ops);
  real M = real(ops);
  
  real n = current_frame;
  real sample_time;
  
  if (dt < 0) {
    sample_time = T - (T - T0) / M * n;
    if (t <= (sample_time + DOLFIN_EPS))
      save_sample = true;
  } else {
    sample_time = T0 + (T - T0) / M * n;
    if (t >= (sample_time - DOLFIN_EPS))
      save_sample = true;
  }
  
  if (save_sample)
    current_frame += 1;
  
  return (save_sample);
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::InitSolutionVariables()
{
  // Initialize matrices
  A_pre = new SparseMatrix();
  A_vel = new SparseMatrix();

  // Initialize solution vectors
  u = new Vector(no_nodes*noeq_vel);
  p = new Vector(no_nodes*noeq_pre);

  upTS = new Vector(no_nodes*noeq_vel); 
  ppTS = new Vector(no_nodes*noeq_pre); 
  upNL = new Vector(no_nodes*noeq_vel); 
  ppNL = new Vector(no_nodes*noeq_pre); 

  upTS->SetToConstant(0.0);
  ppTS->SetToConstant(0.0);
  upNL->SetToConstant(0.0);
  upNL->SetToConstant(0.0);

  print_vector = new Vector(print_no_nodes*noeq);
  
  // Initialize projections (for subgrid modeling)
  u_fine = new Vector(1); 
  p_fine = new Vector(1); 
  u_coarse = new Vector(1); 
  p_coarse = new Vector(1); 

  // FIXME: Should only be initialized if necessary
  u_fine->Resize(no_nodes*noeq_vel); 
  p_fine->Resize(no_nodes*noeq_pre); 
  u_coarse->Resize(no_nodes*noeq_vel); 
  p_coarse->Resize(no_nodes*noeq_pre); 
  
  u_fine->SetToConstant(0.0);
  p_fine->SetToConstant(0.0);
  u_coarse->SetToConstant(0.0);
  p_coarse->SetToConstant(0.0);
  
  tau = new Vector;
  if ( compute_reynolds_stresses ){
    tau->Resize(no_nodes*6);
  }
  
  tau->SetToConstant(0.0);
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::DeleteSolutionVariables()
{
  // Delete matrices 
  delete A_pre;
  delete A_vel;

  // Delete solution vectors
  delete u;
  delete p;
  delete upTS;
  delete ppTS;
  delete upNL;
  delete ppNL;
  delete print_vector;
  
  if ( compute_projections ){
    delete u_fine;
    delete p_fine;
    delete u_coarse;
    delete p_coarse;
  }

  delete tau; 
}
//-----------------------------------------------------------------------------
void ProblemNS::Reset()
{
  // Reset gdisp
  progress = 0.0;
  
  // Reset OpenDX objects
  //opendx_residuals->Reset();

  // Set output frame to zero 
  current_frame = 0;

  // Reset Matlab file
  if ( matlab_data ) matlab_data->Reset();

  // Set tolerances for solution method
  SetTolerances();

  // Initialize subgrid modeling utilities
  SetSubgridModelingUtilities();

  // Update output settings
  settings->Get("write residuals", &write_residuals);
  settings->Get("write data", &write_data);

  settings->Get("start time", &T0);
  settings->Get("final time", &T);

  // Update internal variables 
  no_nodes = grid->GetNoNodes();
  no_cells = grid->GetNoCells();

  print_no_nodes = grid->GetNoNodes();
  print_no_cells = grid->GetNoCells();
  
  A_vel_assembled = false;
  A_pre_assembled = false;
}
//-----------------------------------------------------------------------------
bool ProblemNS::CheckIfStationary()
{
  // Check if the solution is stationary 
  bool stationary_solution = false;

  real maxnorm_du = 0.0;
  for (int i = 0; i < no_nodes; i++) {
    if (fabs(u->Get(i)-upTS->Get(i)) > maxnorm_du)
      maxnorm_du = fabs(u->Get(i) - upTS->Get(i));
  }
  if (maxnorm_du / dt < stat_tol) {
    display->Message(0,"Stationary limit reached, at time t = %.4f, max du/dt = %1.4e",t,maxnorm_du/dt);
    stationary_solution = true;
  }

  return stationary_solution;
}
//----------------------------------------------------------------------------------------------------
bool ProblemNS::CheckNonLinearConvergence(real pre_res_norm_2, real pre_res_norm_inf, 
					  real vel_res_norm_2, real vel_res_norm_inf )
{
  bool non_linear_convergence = false;

  NL_res_norm_2 = sqrt( sqr(pre_res_norm_2) + sqr(vel_res_norm_2) );
  if ( pre_res_norm_inf > vel_res_norm_inf ) NL_res_norm_inf = pre_res_norm_inf;
  else NL_res_norm_inf = vel_res_norm_inf;

  display->Message(4,"l2-norm of NL-residuals at NL iteration %i: ",NL_it+1);
  display->Message(4,"%1.4e (momentum equation)",vel_res_norm_2);
  display->Message(4,"%1.4e (continuity equation)",pre_res_norm_2);
  display->Message(4,"%1.4e (total l2-residual)",NL_res_norm_2);
  display->Message(4,"%1.4e (total max-residual)",NL_res_norm_inf);
    
  //display->Value("Non linear residual",type_real,NL_res_norm_2);
  display->Status(2,"Non linear residual: %1.4e",NL_res_norm_2);
  
  if (  NL_res_norm_2 < NL_tol ) non_linear_convergence = true;
  if ( (NL_res_norm_2 > NL_max_tol) && (NL_it == (max_no_NL_iter-1)) ){
    display->Error("NL residual too large");
  }
  
  return non_linear_convergence;
}
//----------------------------------------------------------------------------------------------------
bool ProblemNS::CheckUPConvergence(real pre_res_norm_2, real pre_res_norm_inf, 
					  real vel_res_norm_2, real vel_res_norm_inf )
{
  bool UP_convergence = false;
  
  UP_res_norm_2 = sqrt( sqr(pre_res_norm_2) + sqr(vel_res_norm_2) );
  if ( pre_res_norm_inf > vel_res_norm_inf ) UP_res_norm_inf = pre_res_norm_inf;
  else UP_res_norm_inf = vel_res_norm_inf;	  

  display->Message(4,"l2-norm of UP-residuals at UP iteration %i: ",UP_it+1);
  display->Message(4,"%1.4e (momentum equation)",vel_res_norm_2);
  display->Message(4,"%1.4e (continuity equation)",pre_res_norm_2);
  display->Message(4,"%1.4e (total l2-residual)",UP_res_norm_2);
  display->Message(4,"%1.4e (total max-residual)",UP_res_norm_inf);

  //display->Value("UP residual",type_real,NL_res_norm_2);
  display->Status(2,"UP residual: %1.4e",UP_res_norm_2);

  if ( UP_res_norm_2 < UP_tol ) UP_convergence = true;
  if ( (UP_res_norm_2 > UP_max_tol) && (UP_it == (max_no_UP_iter-1)) ){
    display->Error("UP residual too large");
  }
  
  return UP_convergence;
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::SetSubgridModelingUtilities()
{
  settings->Get("compute reynolds stresses",&compute_reynolds_stresses);
  settings->Get("write reynolds stresses",&write_reynolds_stresses);

  subgrid_model_on_divergence_form = true;

  // N = (sqr(nsd)+nsd)/2;
  if ( subgrid_model_on_divergence_form ) N = 6; 
  // N =  sqr(nsd);
  else N = 9;             
  // 3 levels 
  // N *= 3;
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::SetInitialData()
{
  display->Status(1,"Set initial conditions");

  initial_data = zero_initial_data;
  
  Point *pnt;
  u->SetToConstant(0.0);
  p->SetToConstant(0.0);
  
  char tmp[256];
  real x,y,z,pert,noRolls;
  real pi = DOLFIN_PI;

  switch ( initial_data ){ 
  case zero_initial_data: 
    break;
  case white_noise: 
    for (int i=0; i < no_nodes*noeq_vel; i++ ){
      u->Set(i,0.1 * 2.0*(drand48() - 0.5));
    }
    break;
  case poiseuille_flow: 
    if (nsd != 3) display->InternalError("ProlemNS::SetInitialData()","Only implemented for 3d");
    pert = 0.0;
    noRolls = 1.0;
    for (int i=0; i < no_nodes; i++ ){
      pnt = grid->GetNode(i)->GetCoord();
      x = real(pnt->x);
      y = real(pnt->y); 
      z = real(pnt->z);
      u->Set(i*noeq_vel+0,16.0*y*(1.0-y)*z*(1-z));
      u->Set(i*noeq_vel+1,pert * (   sin(noRolls*2.0*pi*y) * cos(noRolls*pi*z) ));
      u->Set(i*noeq_vel+2,pert * ( - cos(noRolls*2.0*pi*y) * sin(noRolls*pi*z) ));
    }
    break;
  case couette_flow:
    if (nsd != 3) display->InternalError("ProlemNS::SetInitialData()","Only implemented for 3d");
    pert = 0.0;
    for (int i=0; i < no_nodes; i++ ){
      pnt = grid->GetNode(i)->GetCoord();
      x = real(pnt->x);
      y = real(pnt->y); 
      z = real(pnt->z);
      u->Set(i*noeq_vel+0,2.0*y-1.0);
      u->Set(i*noeq_vel+1,pert * (   sin(2.0*pi*y) * cos(pi*z) ));
      u->Set(i*noeq_vel+2,pert * ( - cos(2.0*pi*y) * sin(pi*z) ));
    }
    break;
  default:
    display->Error("The initial condition is not implemented.");
  }
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::WriteData()
{
  /*
  int el,nod,i;

  // Initializing Fields
  Field Ufld_1(grid);
  Field Ufld_2(grid);
  Field Ufld_3(grid);
  Field Pfld(grid);
  
  Ufld_1.InitByNodes(U,0);
  Ufld_2.InitByNodes(U,1);
  Ufld_3.InitByNodes(U,2);
  Pfld.InitByNodes(U,3);

  // Compute norms of solution
  real L1_norm_U   = Ufld_1.Lp_norm(1) + Ufld_2.Lp_norm(1) + Ufld_3.Lp_norm(1);
  real L2_norm_U   = sqrt( sqr(Ufld_1.Lp_norm(2)) + sqr(Ufld_2.Lp_norm(2)) + sqr(Ufld_3.Lp_norm(2)) );
  real Lmax_norm_U = max( max(Ufld_1.Lmax_norm(),Ufld_2.Lmax_norm()) , Ufld_3.Lmax_norm() );

  real W11_seminorm_U   = Ufld_1.Wkp_seminorm(1,1) + Ufld_2.Wkp_seminorm(1,1) + Ufld_3.Wkp_seminorm(1,1);
  real W12_seminorm_U   = sqrt( sqr(Ufld_1.Wkp_seminorm(1,2)) + sqr(Ufld_2.Wkp_seminorm(1,2)) + sqr(Ufld_3.Wkp_seminorm(1,2)) );
  real W1max_seminorm_U = max( max(Ufld_1.Wkmax_seminorm(1),Ufld_2.Wkmax_seminorm(1)), Ufld_3.Wkmax_seminorm(1) );

  real W11_norm_U   = L1_norm_U + W11_seminorm_U;
  real W12_norm_U   = sqrt( sqr(L2_norm_U) + sqr(W12_seminorm_U) );
  real W1max_norm_U = max( Lmax_norm_U, W1max_seminorm_U );

  real W21_seminorm_U,W22_seminorm_U,W2max_seminorm_U,W21_norm_U,W22_norm_U,W2max_norm_U;
  W21_seminorm_U = W22_seminorm_U = W2max_seminorm_U = W21_norm_U = W22_norm_U = W2max_norm_U = 0.0;
  
	// W21_seminorm_U   = Ufld_1.Wkp_seminorm(2,1) + Ufld_2.Wkp_seminorm(2,1) + Ufld_3.Wkp_seminorm(2,1);
	// W22_seminorm_U   = sqrt( sqr(Ufld_1.Wkp_seminorm(2,2)) + sqr(Ufld_2.Wkp_seminorm(2,2)) + sqr(Ufld_3.Wkp_seminorm(2,2)) );
	// W2max_seminorm_U = max( max(Ufld_1.Wkmax_seminorm(2),Ufld_2.Wkmax_seminorm(2)), Ufld_3.Wkmax_seminorm(2) );

	// W21_norm_U   = W11_norm_U + W21_seminorm_U;
	// W22_norm_U   = sqrt( sqr(W12_norm_U) + sqr(W22_seminorm_U) );
	// W2max_norm_U = max( W1max_norm_U, W2max_seminorm_U );

  real L1_norm_P   = Pfld.Lp_norm(1);
  real L2_norm_P   = Pfld.Lp_norm(2);
  real Lmax_norm_P = Pfld.Lmax_norm();

  real W11_seminorm_P   = Pfld.Wkp_seminorm(1,1);
  real W12_seminorm_P   = Pfld.Wkp_seminorm(1,2);
  real W1max_seminorm_P = Pfld.Wkmax_seminorm(1);

  real W11_norm_P   = L1_norm_P + W11_seminorm_P;
  real W12_norm_P   = sqrt( sqr(L2_norm_P) + sqr(W12_seminorm_P) );
  real W1max_norm_P = max( Lmax_norm_P, W1max_seminorm_P );

  real W21_seminorm_P,W22_seminorm_P,W2max_seminorm_P,W21_norm_P,W22_norm_P,W2max_norm_P;
  W21_seminorm_P = W22_seminorm_P = W2max_seminorm_P = W21_norm_P = W22_norm_P = W2max_norm_P = 0.0;

	// W21_seminorm_P   = Pfld.Wkp_seminorm(2,1);
	// W22_seminorm_P   = Pfld.Wkp_seminorm(2,2);
	// W2max_seminorm_P = Pfld.Wkmax_seminorm(2);

	// W21_norm_P   = W11_norm_P + W21_seminorm_P;
	// W22_norm_P   = sqrt( sqr(W12_norm_P) + sqr(W22_seminorm_P) );
	// W2max_norm_P = max( W1max_norm_P, W2max_seminorm_P );

  // Compute the divergence of the velocity field 
  MV_Vector<real> divU(no_cells);
  for ( el=0; el < no_cells; el++ ) divU(el) = Ufld_1.EvalGradient(el,0) + Ufld_2.EvalGradient(el,1) + Ufld_3.EvalGradient(el,2);
    
  Field Ufld_div(grid);
  Ufld_div.InitByElms(&divU);
  
  real L1_norm_divU   = Ufld_div.Lp_norm(1);
  real L2_norm_divU   = Ufld_div.Lp_norm(2);
  real Lmax_norm_divU = Ufld_div.Lmax_norm();

  // Compute time derivative   
  MV_ColMat<real> Udot(no_nodes,(*U).size(1));
  for ( nod=0; nod < no_nodes; nod++ ){
    for ( i=0; i < (*U).size(1); i++ ) Udot(nod,i) = ( (*U)(nod,i) - (*Uprev)(nod,i) ) / dt; 
  }
  Field Udotfld_1(grid);
  Field Udotfld_2(grid);
  Field Udotfld_3(grid);
  Field Pdotfld(grid);
  Udotfld_1.InitByNodes(&Udot,0);
  Udotfld_2.InitByNodes(&Udot,1);
  Udotfld_3.InitByNodes(&Udot,2);
  Pdotfld.InitByNodes(&Udot,3);
  
  real L1_norm_Udot   = Udotfld_1.Lp_norm(1) + Udotfld_2.Lp_norm(1) + Udotfld_3.Lp_norm(1);
  real L2_norm_Udot   = sqrt( sqr(Udotfld_1.Lp_norm(2)) + sqr(Udotfld_2.Lp_norm(2)) + sqr(Udotfld_3.Lp_norm(2)) );
  real Lmax_norm_Udot = max( max(Udotfld_1.Lmax_norm(),Udotfld_2.Lmax_norm()) , Udotfld_3.Lmax_norm() );

  real L1_norm_Pdot   = Pdotfld.Lp_norm(1);
  real L2_norm_Pdot   = Pdotfld.Lp_norm(2);
  real Lmax_norm_Pdot = Pdotfld.Lmax_norm();

  // Write norms of solution to matlab file
  matlab_data->SetTime(t);

  matlab_data->Set(0,L1_norm_U);
  matlab_data->Set(1,L2_norm_U);
  matlab_data->Set(2,Lmax_norm_U);
  matlab_data->Set(3,W11_seminorm_U);
  matlab_data->Set(4,W12_seminorm_U);
  matlab_data->Set(5,W1max_seminorm_U);
  matlab_data->Set(6,W11_norm_U);
  matlab_data->Set(7,W12_norm_U);
  matlab_data->Set(8,W1max_norm_U);
  matlab_data->Set(9,W21_seminorm_U);
  matlab_data->Set(10,W22_seminorm_U);
  matlab_data->Set(11,W2max_seminorm_U);
  matlab_data->Set(12,W21_norm_U);
  matlab_data->Set(13,W22_norm_U);
  matlab_data->Set(14,W2max_norm_U);
  matlab_data->Set(15,L1_norm_P);
  matlab_data->Set(16,L2_norm_P);
  matlab_data->Set(17,Lmax_norm_P);
  matlab_data->Set(18,W11_seminorm_P);
  matlab_data->Set(19,W12_seminorm_P);
  matlab_data->Set(20,W1max_seminorm_P);
  matlab_data->Set(21,W11_norm_P);
  matlab_data->Set(22,W12_norm_P);
  matlab_data->Set(23,W1max_norm_P);
  matlab_data->Set(24,W21_seminorm_P);
  matlab_data->Set(25,W22_seminorm_P);
  matlab_data->Set(26,W2max_seminorm_P);
  matlab_data->Set(27,W21_norm_P);
  matlab_data->Set(28,W22_norm_P);
  matlab_data->Set(29,W2max_norm_P);
  matlab_data->Set(30,L1_norm_divU);
  matlab_data->Set(31,L2_norm_divU);
  matlab_data->Set(32,Lmax_norm_divU);
  matlab_data->Set(33,L1_norm_Udot);
  matlab_data->Set(34,L2_norm_Udot);
  matlab_data->Set(35,Lmax_norm_Udot);
  matlab_data->Set(36,L1_norm_Pdot);
  matlab_data->Set(37,L2_norm_Pdot);
  matlab_data->Set(38,Lmax_norm_Pdot);
  matlab_data->Save();

  */
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::CompContResidual()
{
  /*
  int nod,el,sd;

  // R1 = | u_t + u*Du + Dp - f |  
  // R2 = max jump in gradient over element boundaries divided by h
  // R3 = time discontinuity (=0 for continuous time integration)
  // R4 = | div u |

  // Computing the R2 residual is expensive due to the approximation of second derivatives
  bool compute_R2 = false;

  //neighbor = &((*GRID)(compGrid).getNeighbor());
  //if ( compute_R2 ){
  //  if ( neighbor->elementSize() == 0 ) neighbor->init( (*GRID)(compGrid), false, false, true );
  // }


  Grid *grid = grid;
  
  Field Ufld_1(grid);
  Field Ufld_2(grid);
  Field Ufld_3(grid);
  Field Pfld(grid);
  
  Ufld_1.InitByNodes(U,0);
  Ufld_2.InitByNodes(U,1);
  Ufld_3.InitByNodes(U,2);
  Pfld.InitByNodes(U,3);

  MV_ColMat<real> Udot(no_nodes,nsd);
  for ( nod=0; nod < no_nodes; nod++ ){
    for ( sd=0; sd < nsd; sd++ ) Udot(nod,sd) = ( (*U)(nod,sd) - (*Uprev)(nod,sd) ) / dt; 
  }
  Field Udotfld_1(grid);
  Field Udotfld_2(grid);
  Field Udotfld_3(grid);
  Udotfld_1.InitByNodes(&Udot,0);
  Udotfld_2.InitByNodes(&Udot,1);
  Udotfld_3.InitByNodes(&Udot,2);
  
  // R3=0 for continuous time integration
  MV_ColMat<real> R1(no_cells,nsd);
  MV_Vector<real> R2(no_cells);
  //MV_ColMat<real> R3(no_cells,nsd);
  MV_Vector<real> R4(no_cells);

  Field Ffld_1(grid);
  Field Ffld_2(grid);
  Field Ffld_3(grid);
  real zero = 0.0;
  Ffld_1.InitByConstant(&zero);
  Ffld_2.InitByConstant(&zero);
  Ffld_3.InitByConstant(&zero);
    
  for ( el=0; el < no_cells; el++ ){
    R1(el,0) = fabs( Udotfld_1.Eval(el) + Ufld_1.Eval(el)*Ufld_1.EvalGradient(el,0) + Ufld_2.Eval(el)*Ufld_1.EvalGradient(el,1) 
							+ Ufld_3.Eval(el)*Ufld_1.EvalGradient(el,2) + Pfld.EvalGradient(el,0) - Ffld_1.Eval(el) );   
    R1(el,1) = fabs( Udotfld_2.Eval(el) + Ufld_1.Eval(el)*Ufld_2.EvalGradient(el,0) + Ufld_2.Eval(el)*Ufld_2.EvalGradient(el,1) 
							+ Ufld_3.Eval(el)*Ufld_2.EvalGradient(el,2) + Pfld.EvalGradient(el,1) - Ffld_2.Eval(el) );   
    R1(el,2) = fabs( Udotfld_3.Eval(el) + Ufld_1.Eval(el)*Ufld_3.EvalGradient(el,0) + Ufld_2.Eval(el)*Ufld_3.EvalGradient(el,1) 
							+ Ufld_3.Eval(el)*Ufld_3.EvalGradient(el,2) + Pfld.EvalGradient(el,2) - Ffld_3.Eval(el) );   
  }
  Field R1fld_1(grid);
  Field R1fld_2(grid);
  Field R1fld_3(grid);
  R1fld_1.InitByElms(&R1,0);
  R1fld_2.InitByElms(&R1,1);
  R1fld_3.InitByElms(&R1,2);

  // Compute norms of R1 residual
  real L1_norm_R1   = R1fld_1.Lp_norm(1) + R1fld_2.Lp_norm(1) + R1fld_3.Lp_norm(1);
  real L2_norm_R1   = sqrt( sqr(R1fld_1.Lp_norm(2)) + sqr(R1fld_2.Lp_norm(2)) + sqr(R1fld_3.Lp_norm(2)) );
  real Lmax_norm_R1 = max( max(R1fld_1.Lmax_norm(),R1fld_2.Lmax_norm()) , R1fld_3.Lmax_norm() );

  // Compute norms of R2 residual
  real L1_norm_R2,L2_norm_R2,Lmax_norm_R2;
  if ( compute_R2 ){
	 
    MV_Vector<real> locGrad(nsd);
    MV_Vector<real> neighborGrad(nsd);
	 
    int i,el_start,el_stop,noNeighbors;
	 Cell *c;
	 
    for ( el=0; el < no_cells; el++ ){

		c = grid->GetCell(el);
		
      R2(el) = 0.0;
      Ufld_1.EvalGradient(el,locGrad);
		
      for (int i=0;i<c->GetNoCellNeighbors();i++){
		  Ufld_1.EvalGradient(c->GetCellNeighbor(i),neighborGrad);
		  for (sd=0;sd<nsd;sd++)
			 if ( fabs(locGrad(sd) - neighborGrad(sd)) > R2(el) )
				R2(el) = fabs(locGrad(sd) - neighborGrad(sd));
      }
		
      Ufld_2.EvalGradient(el,locGrad);
		
      for (int i=0;i<c->GetNoCellNeighbors();i++){
		  Ufld_2.EvalGradient(c->GetCellNeighbor(i),neighborGrad);
		  for ( sd=0; sd < nsd; sd++ ){
			 if ( fabs(locGrad(sd) - neighborGrad(sd)) > R2(el) ) R2(el) = fabs(locGrad(sd) - neighborGrad(sd));
		  }
      }
		
      Ufld_3.EvalGradient(el,locGrad);
		
      for (int i=0;i<c->GetNoCellNeighbors();i++){
		  Ufld_3.EvalGradient(c->GetCellNeighbor(i),neighborGrad);
		  for ( sd=0; sd < nsd; sd++ ){
			 if ( fabs(locGrad(sd) - neighborGrad(sd)) > R2(el) ) R2(el) = fabs(locGrad(sd) - neighborGrad(sd));
		  }
      }
		
      R2(el) /= ( reynolds_number * LocalMeshSize(el) );
    }
    Field R2fld(grid);
    R2fld.InitByElms(&R2);

    L1_norm_R2   = R2fld.Lp_norm(1);
    L2_norm_R2   = R2fld.Lp_norm(2);
    Lmax_norm_R2 = R2fld.Lmax_norm();
  }
  
  // Compute norms of R4 residual
  for ( el=0; el < no_cells; el++ ) R4(el) = Ufld_1.EvalGradient(el,0) + Ufld_2.EvalGradient(el,1) + Ufld_3.EvalGradient(el,2);
  Field R4fld(grid);
  R4fld.InitByElms(&R4);
  
  real L1_norm_R4   = R4fld.Lp_norm(1);
  real L2_norm_R4   = R4fld.Lp_norm(2);
  real Lmax_norm_R4 = R4fld.Lmax_norm();
  
  if ( !compute_R2 ){
    L1_norm_R2   = 0.0;
    L2_norm_R2   = 0.0;
    Lmax_norm_R2 = 0.0;
  }
  real L1_norm_R3   = 0.0;
  real L2_norm_R3   = 0.0;
  real Lmax_norm_R3 = 0.0;

  // Write norms to matlab file
  matlab_data->SetTime(t);

  matlab_data->Set(no_data_entries+0,L1_norm_R1);
  matlab_data->Set(no_data_entries+1,L2_norm_R1);
  matlab_data->Set(no_data_entries+2,Lmax_norm_R1);
  matlab_data->Set(no_data_entries+3,L1_norm_R2);
  matlab_data->Set(no_data_entries+4,L2_norm_R2);
  matlab_data->Set(no_data_entries+5,Lmax_norm_R2);
  matlab_data->Set(no_data_entries+6,L1_norm_R3);
  matlab_data->Set(no_data_entries+7,L2_norm_R3);
  matlab_data->Set(no_data_entries+8,Lmax_norm_R3);
  matlab_data->Set(no_data_entries+9,L1_norm_R4);
  matlab_data->Set(no_data_entries+10,L2_norm_R4);
  matlab_data->Set(no_data_entries+11,Lmax_norm_R4);
  matlab_data->Save();

  // Write element representation of residual
  contResidual.newsize(no_cells);
  for ( el=0; el < no_cells; el++ ) contResidual(el) = R1(el,0) + R1(el,1) + R1(el,2) + R4(el);
  if ( compute_R2 ){
    for ( el=0; el < no_cells; el++ ) contResidual(el) += R2(el);
  }    

  
  // Write nodal representation of residual 
  //MV_Vector<real> R1_1_vec;
  //MV_Vector<real> R1_2_vec;
  //MV_Vector<real> R1_3_vec;
  //MV_Vector<real> R2_vec;
  //MV_Vector<real> R4_vec;
  //R1fld_1.getNodalVector(R1_1_vec);
  //R1fld_2.getNodalVector(R1_2_vec);
  //R1fld_3.getNodalVector(R1_3_vec);
  //R2fld.getNodalVector(R2_vec);
  //R4fld.getNodalVector(R4_vec);
  //MV_ColMat<real> printRes(printNoNodes,5);
  //for ( i=0; i < printNoNodes; i++ ){
  //printRes(i,0) = R1_1_vec(i);
  //printRes(i,1) = R1_2_vec(i);
  //printRes(i,2) = R1_3_vec(i);
  //printRes(i,3) = R2_vec(i);
  //printRes(i,4) = R4_vec(i);
  //}
  //opendx_residuals->AddFrame(&(*GRID)(printGrid),&printRes,t);
  */
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::WritePertubationData()
{
  /*
  int nod,el;
  real x,y,z;
  Point p;
  Grid *grid = grid;
  
  // Write pertubation data to file
  Field Ufld_1(grid);
  Field Ufld_2(grid);
  Field Ufld_3(grid);
  
  Ufld_1.InitByNodes(U,0);
  Ufld_2.InitByNodes(U,1);
  Ufld_3.InitByNodes(U,2);

  Field Upertfld_1(grid);
  Field Upertfld_2(grid);
  Field Upertfld_3(grid);
  
  MV_Vector<real> Utmp(no_nodes);
  
  for (int i=0;i<no_nodes;i++){
	 p = grid->GetNode(i)->GetCoord();
    y = real(p.y);
    z = real(p.z);
    if ( pert_poiseuille ) Utmp(nod) = (*U)(nod,0) - ( 16.0*y*(1.0-y)*z*(1-z) );
    if ( pert_couette )    Utmp(nod) = (*U)(nod,0) - ( 2.0*y-1.0 );
  }
  Upertfld_1.InitByNodes(&Utmp);
  Upertfld_2.InitByNodes(U,1);
  Upertfld_3.InitByNodes(U,2);

  real L2_norm_U1pert, L2_norm_U2pert, L2_norm_U3pert;
  real Lmax_norm_U1pert, Lmax_norm_U2pert, Lmax_norm_U3pert;
  real L2_norm_D1U1, L2_norm_D2U1, L2_norm_D3U1;
  real L2_norm_D1U2, L2_norm_D2U2, L2_norm_D3U2;
  real L2_norm_D1U3, L2_norm_D2U3, L2_norm_D3U3;
  real Lmax_norm_D1U1, Lmax_norm_D2U1, Lmax_norm_D3U1;
  real Lmax_norm_D1U2, Lmax_norm_D2U2, Lmax_norm_D3U2;
  real Lmax_norm_D1U3, Lmax_norm_D2U3, Lmax_norm_D3U3;

  L2_norm_U1pert   = Upertfld_1.Lp_norm(2);
  L2_norm_U2pert   = Upertfld_2.Lp_norm(2);
  L2_norm_U3pert   = Upertfld_3.Lp_norm(2);
  Lmax_norm_U1pert = Upertfld_1.Lmax_norm();
  Lmax_norm_U2pert = Upertfld_2.Lmax_norm();
  Lmax_norm_U3pert = Upertfld_3.Lmax_norm();
  
  Field Ugradfld(grid);
  Utmp.newsize(no_cells);
  
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_1.EvalGradient(el,0);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D1U1  = Ugradfld.Lp_norm(2);
  Lmax_norm_D1U1 = Ugradfld.Lmax_norm();
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_1.EvalGradient(el,1);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D2U1  = Ugradfld.Lp_norm(2);
  Lmax_norm_D2U1 = Ugradfld.Lmax_norm();
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_1.EvalGradient(el,2);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D3U1  = Ugradfld.Lp_norm(2);
  Lmax_norm_D3U1 = Ugradfld.Lmax_norm();
  
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_2.EvalGradient(el,0);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D1U2  = Ugradfld.Lp_norm(2);
  Lmax_norm_D1U2 = Ugradfld.Lmax_norm();
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_2.EvalGradient(el,1);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D2U2  = Ugradfld.Lp_norm(2);
  Lmax_norm_D2U2 = Ugradfld.Lmax_norm();
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_2.EvalGradient(el,2);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D3U2  = Ugradfld.Lp_norm(2);
  Lmax_norm_D3U2 = Ugradfld.Lmax_norm();
  
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_3.EvalGradient(el,0);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D1U3  = Ugradfld.Lp_norm(2);
  Lmax_norm_D1U3 = Ugradfld.Lmax_norm();
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_3.EvalGradient(el,1);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D2U3  = Ugradfld.Lp_norm(2);
  Lmax_norm_D2U3 = Ugradfld.Lmax_norm();
  for ( el=0; el < no_cells; el++ ) Utmp(el) = Ufld_3.EvalGradient(el,2);
  Ugradfld.InitByElms(&Utmp);
  L2_norm_D3U3  = Ugradfld.Lp_norm(2);
  Lmax_norm_D3U3 = Ugradfld.Lmax_norm();
  
  // Write data to matlab file 
  matlab_data->SetTime(t);

  matlab_data->Set(no_data_entries+no_residual_entries+0,L2_norm_U1pert);
  matlab_data->Set(no_data_entries+no_residual_entries+1,Lmax_norm_U1pert);
  matlab_data->Set(no_data_entries+no_residual_entries+2,L2_norm_U2pert);
  matlab_data->Set(no_data_entries+no_residual_entries+3,Lmax_norm_U2pert);
  matlab_data->Set(no_data_entries+no_residual_entries+4,L2_norm_U3pert);
  matlab_data->Set(no_data_entries+no_residual_entries+5,Lmax_norm_U3pert);
  matlab_data->Set(no_data_entries+no_residual_entries+6,L2_norm_D1U1);
  matlab_data->Set(no_data_entries+no_residual_entries+7,Lmax_norm_D1U1);
  matlab_data->Set(no_data_entries+no_residual_entries+8,L2_norm_D2U1);
  matlab_data->Set(no_data_entries+no_residual_entries+9,Lmax_norm_D2U1);
  matlab_data->Set(no_data_entries+no_residual_entries+10,L2_norm_D3U1);
  matlab_data->Set(no_data_entries+no_residual_entries+11,Lmax_norm_D3U1);
  matlab_data->Set(no_data_entries+no_residual_entries+12,L2_norm_D1U2);
  matlab_data->Set(no_data_entries+no_residual_entries+13,Lmax_norm_D1U2);
  matlab_data->Set(no_data_entries+no_residual_entries+14,L2_norm_D2U2);
  matlab_data->Set(no_data_entries+no_residual_entries+15,Lmax_norm_D2U2);
  matlab_data->Set(no_data_entries+no_residual_entries+16,L2_norm_D3U2);
  matlab_data->Set(no_data_entries+no_residual_entries+17,Lmax_norm_D3U2);
  matlab_data->Set(no_data_entries+no_residual_entries+18,L2_norm_D1U3);
  matlab_data->Set(no_data_entries+no_residual_entries+19,Lmax_norm_D1U3);
  matlab_data->Set(no_data_entries+no_residual_entries+20,L2_norm_D2U3);
  matlab_data->Set(no_data_entries+no_residual_entries+21,Lmax_norm_D2U3);
  matlab_data->Set(no_data_entries+no_residual_entries+22,L2_norm_D3U3);
  matlab_data->Set(no_data_entries+no_residual_entries+23,Lmax_norm_D3U3);
  matlab_data->Save();
  */
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::WriteReynoldsStresses()
{
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::ComputeFunctionals()
{
  /*
  int nod,el,sd;

  Field Ufld_1(grid);
  Field Ufld_2(grid);
  Field Ufld_3(grid);
  Field Pfld(grid);
  
  Ufld_1.InitByNodes(U,0);
  Ufld_2.InitByNodes(U,1);
  Ufld_3.InitByNodes(U,2);
  Pfld.InitByNodes(U,3);

  MV_ColMat<real> Udot(no_nodes,nsd);
  for ( nod=0; nod < no_nodes; nod++ ){
    for ( sd=0; sd < nsd; sd++ ) Udot(nod,sd) = ( (*U)(nod,sd) - (*Uprev)(nod,sd) ) / dt; 
  }
  Field Udotfld_1(grid);
  Field Udotfld_2(grid);
  Field Udotfld_3(grid);
  Udotfld_1.InitByNodes(&Udot,0);
  Udotfld_2.InitByNodes(&Udot,1);
  Udotfld_3.InitByNodes(&Udot,2);
  
  Field Ffld_1(grid);
  Field Ffld_2(grid);
  Field Ffld_3(grid);
  real zero = 0.0;
  Ffld_1.InitByConstant(&zero);
  Ffld_2.InitByConstant(&zero);
  Ffld_3.InitByConstant(&zero);

  Field Psi_1(grid);
  Field Psi_2(grid);
  Field Psi_3(grid);
  Psi_1.InitByConstant(&zero);
  Psi_2.InitByConstant(&zero);
  Psi_3.InitByConstant(&zero);
    
  real drag = 0.0;
  for ( el=0; el < no_cells; el++ ){
    drag += ( Udotfld_1.Eval(el) + Ufld_1.Eval(el)*Ufld_1.EvalGradient(el,0) + Ufld_2.Eval(el)*Ufld_1.EvalGradient(el,1) 
				  + Ufld_3.Eval(el)*Ufld_1.EvalGradient(el,2) ) * Psi_1.Eval(el) 
		+ ( Udotfld_2.Eval(el) + Ufld_1.Eval(el)*Ufld_2.EvalGradient(el,0) + Ufld_2.Eval(el)*Ufld_2.EvalGradient(el,1) 
			 + Ufld_3.Eval(el)*Ufld_2.EvalGradient(el,2) ) * Psi_2.Eval(el) 
		+ ( Udotfld_3.Eval(el) + Ufld_1.Eval(el)*Ufld_3.EvalGradient(el,0) + Ufld_2.Eval(el)*Ufld_3.EvalGradient(el,1) 
			 + Ufld_3.Eval(el)*Ufld_3.EvalGradient(el,2) ) * Psi_3.Eval(el) 
		- Pfld.Eval(el) * (Psi_1.EvalGradient(el,0)+Psi_2.EvalGradient(el,1)+Psi_3.EvalGradient(el,2))
		+ (2.0/reynolds_number) * ( Ufld_1.EvalGradient(el,0)*Psi_1.EvalGradient(el,0)  
											 + 0.5*(Ufld_1.EvalGradient(el,1)+Ufld_2.EvalGradient(el,0))*0.5*(Psi_1.EvalGradient(el,1)+Psi_2.EvalGradient(el,0))
											 + 0.5*(Ufld_1.EvalGradient(el,2)+Ufld_3.EvalGradient(el,0))*0.5*(Psi_1.EvalGradient(el,2)+Psi_3.EvalGradient(el,0))
											 + 0.5*(Ufld_2.EvalGradient(el,0)+Ufld_1.EvalGradient(el,1))*0.5*(Psi_2.EvalGradient(el,0)+Psi_1.EvalGradient(el,1))
											 + Ufld_2.EvalGradient(el,1)*Psi_2.EvalGradient(el,1)
											 + 0.5*(Ufld_2.EvalGradient(el,2)+Ufld_3.EvalGradient(el,1))*0.5*(Psi_2.EvalGradient(el,2)+Psi_3.EvalGradient(el,1))
											 + 0.5*(Ufld_3.EvalGradient(el,0)+Ufld_1.EvalGradient(el,2))*0.5*(Psi_3.EvalGradient(el,0)+Psi_1.EvalGradient(el,0))
											 + 0.5*(Ufld_3.EvalGradient(el,1)+Ufld_2.EvalGradient(el,2))*0.5*(Psi_3.EvalGradient(el,1)+Psi_2.EvalGradient(el,0))
											 + Ufld_3.EvalGradient(el,2)*Psi_3.EvalGradient(el,2) )
		- ( Ffld_1.Eval(el)*Psi_1.Eval(el) + Ffld_2.Eval(el)*Psi_2.Eval(el) + Ffld_3.Eval(el)*Psi_3.Eval(el) );   
  }

  cout << "Drag force = " << drag << endl;
  

  // Write norms to matlab file
  //matlab_data->SetTime(t);
  //matlab_data->Set(no_data_entries+0,drag);
  //matlab_data->Save();
  */
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::SetTolerances()
{
  // Set tolerances for iterative methods
  NL_tol   = 1.0e-4;
  UP_tol   = 1.0e-1;
  stat_tol = 1.0e-10;

  NL_max_tol = 1000.0;
  UP_max_tol = 1000.0;

  max_no_UP_iter = 1;
  max_no_NL_iter = 100;  

  UP_conv = true;
  NL_conv = true;
}
//-----------------------------------------------------------------------------
real ProblemNS::GetTimeStep()
{
  return grid_h;
}
//-----------------------------------------------------------------------------
void ProblemNS::GetMeshSize()
{
  // Compute smallest element diameter in grid
  
  real lms = LocalMeshSize(0);
  grid_h = lms;
  for (int el = 1; el < no_cells; el++) {
    lms = LocalMeshSize(el);
    if (lms < grid_h) grid_h = lms;
  }
}
//-----------------------------------------------------------------------------
real ProblemNS::LocalMeshSize(int el)
{
  // Compute element diameter
  
  real gh = 2.0 * grid->GetCell(el)->ComputeCircumRadius(grid);
  
  return gh;
}
//----------------------------------------------------------------------------------------------------
void ProblemNS::SetTurbulentInflow( real trb )
{
  /*
  int ierr;
  int i,j;
  int *bndNodes; 
  int *b_nodes;
  int noBndNodes;

  b_nodes = new int[no_nodes];

  ierr = getBndNodes(b_nodes, &noBndNodes, bndx0, grid);

  turbulence.newsize(noBndNodes,noeq);
  for ( i=0; i < noBndNodes; i++ ){
    for ( j=0; j < noeq; j++ ){
      turbulence(i,j) = trb * 2.0*(drand48() - 0.5);
    }
  }

  delete b_nodes
  */
}
//----------------------------------------------------------------------------------------------------
