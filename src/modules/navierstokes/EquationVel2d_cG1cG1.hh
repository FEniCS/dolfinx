#ifndef __EQUATION_VEL2D_CG1_CG1_HH
#define __EQUATION_VEL2D_CG1_CG1_HH

#include <ShapeFunction.hh>
#include <LocalField.hh>
#include <GlobalField.hh>
#include <SettingsNavierStokes.hh>
#include <EquationSystem.hh>
#include <TriLinSpace.hh>
#include <math.h>

class EquationVel2d_cG1cG1:public EquationSystem {
  
public:
  
  EquationVel2d_cG1cG1() : EquationSystem(2,2) {
	 
    start_vector_component = 0;
    
    AllocateFields(17);
    
    field[0] = &upTS[0];
    field[1] = &upTS[1];
    field[2] = &ppTS;
    
    field[3] = &upNL[0];
    field[4] = &upNL[1];
    field[5] = &ppNL;

    field[6]  = &unow[0];
    field[7]  = &unow[1];
    field[8] = &pnow;
	 
    field[9] = &u_fine[0];
    field[10] = &u_fine[1];
    field[11] = &p_fine[0];
	 
    field[12] = &u_coarse[0];
    field[13] = &u_coarse[1];
    field[14] = &p_coarse[0];
	 
    field[15] = &f[0];
    field[16] = &f[1];
	 
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

    MASS = ( u[0]*v[0] + u[1]*v[1] );
	 
    LAP  = 0.5*nu*( u[0].dx*v[0].dx + u[0].dy*v[0].dy + 
						  u[1].dx*v[1].dx + u[1].dy*v[1].dy );    
	 
    NL   = 0.5 * ( u[0].dx * (U[0]*v[0]) + u[0].dy * (U[1]*v[0]) + 
						 u[1].dx * (U[0]*v[1]) + u[1].dy * (U[1]*v[1]) );
	 
    NLSD = 0.5 * ( u[0].dx*v[0].dx*U0_U0 + u[0].dx*v[0].dy*U0_U1 + 
						 u[0].dy*v[0].dx*U0_U1 + u[0].dy*v[0].dy*U1_U1 +
						 
						 u[1].dx*v[1].dx*U0_U0 + u[1].dx*v[1].dy*U0_U1 +
						 u[1].dy*v[1].dx*U0_U1 + u[1].dy*v[1].dy*U1_U1 ); 
	 
    DIV  = 0.5 * ( u[0].dx + u[1].dy) * (v[0].dx + v[1].dy);
    
    return ( MASS + dt * (LAP + NL + delta1*NLSD + delta2*DIV ) );
  }
  //---------------------------------------------------------------------------
  real IntegrateRHS(ShapeFunction *v){
    
    MASS   = ( upTS[0]*v[0] + upTS[1]*v[1] );
    
    LAP    = 0.5*nu*( upTS[0].dx*v[0].dx + upTS[0].dy*v[0].dy + 
							 upTS[1].dx*v[1].dx + upTS[1].dy*v[1].dy ); 
    
    NL     = 0.5*( upTS[0].dx * (U[0]*v[0]) + upTS[0].dy * (U[1]*v[0]) +
						 upTS[1].dx * (U[0]*v[1]) + upTS[1].dy * (U[1]*v[1]) ); 
    
    NLSD   = 0.5*( upTS[0].dx*v[0].dx*U0_U0 + upTS[0].dx*v[0].dy*U0_U1 + 
						 upTS[0].dy*v[0].dx*U0_U1 + upTS[0].dy*v[0].dy*U1_U1 +
						 
						 upTS[1].dx*v[1].dx*U0_U0 + upTS[1].dx*v[1].dy*U0_U1 +
						 upTS[1].dy*v[1].dx*U0_U1 + upTS[1].dy*v[1].dy*U1_U1 );
	 
    DIV    = 0.5*( upTS[0].dx + upTS[1].dy) * (v[0].dx + v[1].dy);
    
    PRE    = pnow * (v[0].dx + v[1].dy);
    
    PRESD  = 0.5*( pnow.dx*v[0].dx*U[0] + pnow.dx*v[0].dy*U[1] + 
						 pnow.dy*v[1].dx*U[0] + pnow.dy*v[1].dy*U[1] );
	 
    FORCE  = f[0]*v[0] + f[1]*v[1];
    
    FORCESD = 0.5*( v[0].dx*(f[0]*U[0]) + v[0].dy*(f[0]*U[1]) + 
						  v[1].dx*(f[1]*U[0]) + v[1].dy*(f[1]*U[1]) );
    
    return ( MASS + dt * (FORCE - LAP - NL + PRE + delta1*(FORCESD-NLSD-PRESD) - delta2*DIV) );
  }
  //---------------------------------------------------------------------------  
  void UpdateLHS(){
  
    //normU = sqrt( sqr(unow[0].GetMeanValue()) +
    //              sqr(unow[1].GetMeanValue()));
    
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

    U0_U0 = (U[0]*U[0]); U0_U1 = (U[0]*U[1]);
    U1_U1 = (U[1]*U[1]);

    nu = 1.0/Re;
    
    UpdateTurbulenceModel();
    
  }
  //---------------------------------------------------------------------------  
  void UpdateRHS(){

    //normU = sqrt( sqr(unow[0].GetMeanValue()) +
    //              sqr(unow[1].GetMeanValue()));
	 
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

    U0_U0 = (U[0]*U[0]); U0_U1 = (U[0]*U[1]);
    U1_U1 = (U[1]*U[1]);
	 
    nu = 1.0/Re;
	 
    UpdateTurbulenceModel();
    
  }
  //---------------------------------------------------------------------------
  void UpdateTurbulenceModel(){

    if ( !turbulence_model )
      return;
    
    S_abs = sqrt( sqr(upNL[0].dx) + 
		  sqr( 0.5*(upNL[0].dy + upNL[1].dx)) + 
		  sqr( 0.5*(upNL[1].dx + upNL[0].dy)) + 
		  sqr( upNL[1].dy) );
    
    if (smagorinsky_eddy_viscosity)
      nu += pow(Cs * h, 2.0) * S_abs;
    
    /*
      if (variational_multiscale_eddy_viscosity) {      
      LAP_VMM = nu_T * ( (u[0].dx - u_coarse[0].dx) * v[0].dx +  
      (u[0].dy - u_coarse[0].dy) * v[0].dy +  
      (u[1].dx - u_coarse[1].dx) * v[1].dx +  
      (u[1].dy - u_coarse[1].dy) * v[1].dy +  
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

  real U0_U0,U0_U1,U1_U1;

  bool variational_multiscale_eddy_viscosity;
  bool smagorinsky_eddy_viscosity;
  bool turbulence_model;

  LocalField u_fine[2];
  LocalField p_fine[1];
  LocalField u_coarse[2];
  LocalField p_coarse[1];
  
  LocalField upTS[2];
  LocalField ppTS;
  LocalField upNL[2];
  LocalField ppNL;

  LocalField unow[2];
  LocalField pnow;
  LocalField f[2];
  
  LocalField U[2];
  
};
  
#endif
