#ifndef __EQUATION_VEL_CG1_CG1_HH
#define __EQUATION_VEL_CG1_CG1_HH

#include <ShapeFunction.hh>
#include <LocalField.hh>
#include <GlobalField.hh>
#include <SettingsNavierStokes.hh>
#include <EquationSystem.hh>
#include <TetLinSpace.hh>
#include <math.h>

class EquationVel_cG1cG1:public EquationSystem {
  
public:
  
  EquationVel_cG1cG1() : EquationSystem(3,3) {
	 
    start_vector_component = 0;
    
    AllocateFields(23);
    
    field[0] = &upTS[0];
    field[1] = &upTS[1];
    field[2] = &upTS[2];
    field[3] = &ppTS;
    
    field[4] = &upNL[0];
    field[5] = &upNL[1];
    field[6] = &upNL[2];
    field[7] = &ppNL;

    field[8]  = &unow[0];
    field[9]  = &unow[1];
    field[10] = &unow[2];
    field[11] = &pnow;
	 
    field[12] = &u_fine[0];
    field[13] = &u_fine[1];
    field[14] = &u_fine[2];
    field[15] = &p_fine[0];
	 
    field[16] = &u_coarse[0];
    field[17] = &u_coarse[1];
    field[18] = &u_coarse[2];
    field[19] = &p_coarse[0];
	 
    field[20] = &f[0];
    field[21] = &f[1];
    field[22] = &f[2];
	 
    C1 = 1.0;
    C2 = 1.0;
	 
    Cs = 0.1;
    
    settings->Get("reynolds number",&Re);
	 
    settings->Get("variational multscale eddy viscosity",
		  &variational_multiscale_eddy_viscosity);
    settings->Get("smagorinsky eddy viscosity",
		  &smagorinsky_eddy_viscosity);
    
    turbulence_model                      = false;
    variational_multiscale_eddy_viscosity = false;
    smagorinsky_eddy_viscosity            = false;
	 
  }
  //---------------------------------------------------------------------------  
  real IntegrateLHS(ShapeFunction *u, ShapeFunction *v){

	 register real u0dx_v0dx = u[0].dx*v[0].dx;
	 register real u0dy_v0dy = u[0].dy*v[0].dy;
	 register real u0dz_v0dz = u[0].dz*v[0].dz;
	 register real u1dx_v1dx = u[1].dx*v[1].dx;
	 register real u1dy_v1dy = u[1].dy*v[1].dy;
	 register real u1dz_v1dz = u[1].dz*v[1].dz;
	 register real u2dx_v2dx = u[2].dx*v[2].dx;
	 register real u2dy_v2dy = u[2].dy*v[2].dy;
	 register real u2dz_v2dz = u[2].dz*v[2].dz;
	 
    MASS = ( u[0]*v[0] + u[1]*v[1] + u[2]*v[2] );

    LAP = 0.5*nu*( u0dx_v0dx + u0dy_v0dy + u0dz_v0dz +
						 u1dx_v1dx + u1dy_v1dy + u1dz_v1dz +
						 u2dx_v2dx + u2dy_v2dy + u2dz_v2dz );

    NL = 0.5*( u[0].dx * (U[0]*v[0]) + u[0].dy * (U[1]*v[0]) + u[0].dz * (U[2]*v[0]) + 
					u[1].dx * (U[0]*v[1]) + u[1].dy * (U[1]*v[1]) + u[1].dz * (U[2]*v[1]) + 
					u[2].dx * (U[0]*v[2]) + u[2].dy * (U[1]*v[2]) + u[2].dz * (U[2]*v[2]) );
    
    NLSD = 0.5*( u0dx_v0dx*U0_U0       + u[0].dx*v[0].dy*U0_U1 + u[0].dx*v[0].dz*U0_U2 + 
					  u[0].dy*v[0].dx*U0_U1 + u0dy_v0dy*U1_U1       + u[0].dy*v[0].dz*U1_U2 +
					  u[0].dz*v[0].dx*U0_U2 + u[0].dz*v[0].dy*U1_U2 + u0dz_v0dz*U2_U2 +
					  
					  u1dx_v1dx*U0_U0       + u[1].dx*v[1].dy*U0_U1 + u[1].dx*v[1].dz*U0_U2 + 
					  u[1].dy*v[1].dx*U0_U1 + u1dy_v1dy*U1_U1           + u[1].dy*v[1].dz*U1_U2 +
					  u[1].dz*v[1].dx*U0_U2 + u[1].dz*v[1].dy*U1_U2 + u1dz_v1dz*U2_U2 +
					  
					  u2dx_v2dx*U0_U0       + u[2].dx*v[2].dy*U0_U1 + u[2].dx*v[2].dz*U0_U2 + 
					  u[2].dy*v[2].dx*U0_U1 + u2dy_v2dy*U1_U1       + u[2].dy*v[2].dz*U1_U2 +
					  u[2].dz*v[2].dx*U0_U2 + u[2].dz*v[2].dy*U1_U2 + u2dz_v2dz*U2_U2 );
	 
    DIV  = 0.5 * ( u[0].dx + u[1].dy + u[2].dz) * (v[0].dx + v[1].dy + v[2].dz);
    
    return ( MASS + dt * (LAP + NL + delta1*NLSD + delta2*DIV ) );
  }
  //---------------------------------------------------------------------------
  real IntegrateRHS(ShapeFunction *v){

	 register real upTS0dx_v0dx = upTS[0].dx*v[0].dx;
	 register real upTS0dy_v0dy = upTS[0].dy*v[0].dy;
	 register real upTS0dz_v0dz = upTS[0].dz*v[0].dz;
	 register real upTS1dx_v1dx = upTS[1].dx*v[1].dx;
	 register real upTS1dy_v1dy = upTS[1].dy*v[1].dy;
	 register real upTS1dz_v1dz = upTS[1].dz*v[1].dz;
	 register real upTS2dx_v2dx = upTS[2].dx*v[2].dx;
	 register real upTS2dy_v2dy = upTS[2].dy*v[2].dy;
	 register real upTS2dz_v2dz = upTS[2].dz*v[2].dz;
	 	 
    MASS = ( upTS[0]*v[0] + upTS[1]*v[1] + upTS[2]*v[2] );
    
    LAP = 0.5*nu*( upTS[0].dx*v[0].dx + upTS[0].dy*v[0].dy + upTS[0].dz*v[0].dz + 
						 upTS[1].dx*v[1].dx + upTS[1].dy*v[1].dy + upTS[1].dz*v[1].dz + 
						 upTS[2].dx*v[2].dx + upTS[2].dy*v[2].dy + upTS[2].dz*v[2].dz );
    
    NL = 0.5*( upTS[0].dx * (U[0]*v[0]) + upTS[0].dy * (U[1]*v[0]) + upTS[0].dz * (U[2]*v[0]) + 
					upTS[1].dx * (U[0]*v[1]) + upTS[1].dy * (U[1]*v[1]) + upTS[1].dz * (U[2]*v[1]) + 
					upTS[2].dx * (U[0]*v[2]) + upTS[2].dy * (U[1]*v[2]) + upTS[2].dz * (U[2]*v[2]) );
    
    NLSD = 0.5*( upTS0dx_v0dx*U0_U0       + upTS[0].dx*v[0].dy*U0_U1 + upTS[0].dx*v[0].dz*U0_U2 + 
					  upTS[0].dy*v[0].dx*U0_U1 + upTS0dy_v0dy*U1_U1       + upTS[0].dy*v[0].dz*U1_U2 +
					  upTS[0].dz*v[0].dx*U0_U2 + upTS[0].dz*v[0].dy*U1_U2 + upTS0dz_v0dz*U2_U2 +
					  
					  upTS1dx_v1dx*U0_U0       + upTS[1].dx*v[1].dy*U0_U1 + upTS[1].dx*v[1].dz*U0_U2 + 
					  upTS[1].dy*v[1].dx*U0_U1 + upTS1dy_v1dy*U1_U1       + upTS[1].dy*v[1].dz*U1_U2 +
					  upTS[1].dz*v[1].dx*U0_U2 + upTS[1].dz*v[1].dy*U1_U2 + upTS1dz_v1dz*U2_U2 +
					  
					  upTS2dx_v2dx*U0_U0       + upTS[2].dx*v[2].dy*U0_U1 + upTS[2].dx*v[2].dz*U0_U2 + 
					  upTS[2].dy*v[2].dx*U0_U1 + upTS2dy_v2dy*U1_U1       + upTS[2].dy*v[2].dz*U1_U2 +
					  upTS[2].dz*v[2].dx*U0_U2 + upTS[2].dz*v[2].dy*U1_U2 + upTS2dz_v2dz*U2_U2 );
	 
    DIV = 0.5*( upTS[0].dx + upTS[1].dy + upTS[2].dz) * (v[0].dx + v[1].dy + v[2].dz);
    
    PRE = pnow * (v[0].dx + v[1].dy + v[2].dz);
    
    PRESD = 0.5*( pnow.dx*v[0].dx*U[0] + pnow.dx*v[0].dy*U[1] + pnow.dx*v[0].dz*U[2] + 
						pnow.dy*v[1].dx*U[0] + pnow.dy*v[1].dy*U[1] + pnow.dy*v[1].dz*U[2] + 
						pnow.dz*v[2].dx*U[0] + pnow.dz*v[2].dy*U[1] + pnow.dz*v[2].dz*U[2] );
    
    FORCE = f[0]*v[0] + f[1]*v[1] + f[2]*v[2];
    
    FORCESD = 0.5*( v[0].dx*(f[0]*U[0]) + v[0].dy*(f[0]*U[1]) + v[0].dz*(f[0]*U[2]) + 
						  v[1].dx*(f[1]*U[0]) + v[1].dy*(f[1]*U[1]) + v[1].dz*(f[1]*U[2]) + 
						  v[2].dx*(f[2]*U[0]) + v[2].dy*(f[2]*U[1]) + v[2].dz*(f[2]*U[2]) );
    
    return ( MASS + dt * (FORCE - LAP - NL + PRE + delta1*(FORCESD-NLSD-PRESD) - delta2*DIV) );
  }
  //---------------------------------------------------------------------------  
  void UpdateLHS(){
  
    //normU = sqrt( sqr(unow[0].GetMeanValue()) +
    //              sqr(unow[1].GetMeanValue()) +
    //              sqr(unow[2].GetMeanValue()));
    
    if (h * Re > 1.0){
      delta1 = C1 * h;
      delta2 = C2 * h;
    }
    else{
      delta1 = C1 * sqr(h);
      delta2 = C2 * sqr(h);
    }
    
    U[0].Mean(upTS[0],upNL[0]);
    U[1].Mean(upTS[1],upNL[1]);
    U[2].Mean(upTS[2],upNL[2]);

    U0_U0 = (U[0]*U[0]); U0_U1 = (U[0]*U[1]); U0_U2 = (U[0]*U[2]);
    U1_U1 = (U[1]*U[1]); U1_U2 = (U[1]*U[2]);
    U2_U2 = (U[2]*U[2]);

    nu = 1.0/Re;
    
    UpdateTurbulenceModel();
    
  }
  //---------------------------------------------------------------------------  
  void UpdateRHS(){

    //normU = sqrt( sqr(unow[0].GetMeanValue()) +
    //              sqr(unow[1].GetMeanValue()) +
    //              sqr(unow[2].GetMeanValue()));
	 
    if (h * Re > 1.0){
      delta1 = C1 * h;
      delta2 = C2 * h;
    }
    else{
      delta1 = C1 * sqr(h);
      delta2 = C2 * sqr(h);
    }
	 
    U[0].Mean(upTS[0],upNL[0]);
    U[1].Mean(upTS[1],upNL[1]);
    U[2].Mean(upTS[2],upNL[2]);
    
    U0_U0 = (U[0]*U[0]); U0_U1 = (U[0]*U[1]); U0_U2 = (U[0]*U[2]);
    U1_U1 = (U[1]*U[1]); U1_U2 = (U[1]*U[2]);
    U2_U2 = (U[2]*U[2]);

    nu = 1.0/Re;
	 
    UpdateTurbulenceModel();
    
  }
  //---------------------------------------------------------------------------
  void UpdateTurbulenceModel(){

    if ( !turbulence_model )
      return;
    
    S_abs = sqrt( sqr(upNL[0].dx) + 
		  sqr( 0.5*(upNL[0].dy + upNL[1].dx)) + 
		  sqr( 0.5*(upNL[0].dz + upNL[2].dx)) + 
		  sqr( 0.5*(upNL[1].dx + upNL[0].dy)) + 
		  sqr( upNL[1].dy) + 
		  sqr( 0.5*(upNL[1].dz + upNL[2].dy)) + 
		  sqr( 0.5*(upNL[2].dx + upNL[0].dz)) + 
		  sqr( 0.5*(upNL[2].dy + upNL[1].dz)) + 
		  sqr(upNL[2].dz) );
    
    if (smagorinsky_eddy_viscosity)
      nu += pow(Cs * h, 2.0) * S_abs;
    
    /*
      if (variational_multiscale_eddy_viscosity) {      
      LAP_VMM = nu_T * ( (u[0].dx - u_coarse[0].dx) * v[0].dx +  
      (u[0].dy - u_coarse[0].dy) * v[0].dy +  
      (u[0].dz - u_coarse[0].dz) * v[0].dz +  
      (u[1].dx - u_coarse[1].dx) * v[1].dx +  
      (u[1].dy - u_coarse[1].dy) * v[1].dy +  
      (u[1].dz - u_coarse[1].dz) * v[1].dz +  
      (u[2].dx - u_coarse[2].dx) * v[2].dx +  
      (u[2].dy - u_coarse[2].dy) * v[2].dy +  
      (u[2].dz - u_coarse[2].dz) * v[2].dz );  
      } else{
      LAP_VMM = 0.0;
      }
    */
    
  }
  //---------------------------------------------------------------------------
  
private:
  
  real delta1, delta2;
  real C1, C2;
  
  real Re;

  real Cs, S_abs, nu, nu_T;

  real MASS, NL, NLSD, LAP, DIV, FORCE, FORCESD, PRE, PRESD, LAP_VMM;

  real U0_U0,U0_U1,U0_U2,U1_U1,U1_U2,U2_U2;

  bool variational_multiscale_eddy_viscosity;
  bool smagorinsky_eddy_viscosity;
  bool turbulence_model;

  LocalField u_fine[3];
  LocalField p_fine[1];
  LocalField u_coarse[3];
  LocalField p_coarse[1];
  
  LocalField upTS[3];
  LocalField ppTS;
  LocalField upNL[3];
  LocalField ppNL;

  LocalField unow[3];
  LocalField pnow;
  LocalField f[3];
  
  LocalField U[3];
  
};
  
#endif
