#ifndef __EQUATION_PRE2D_CG1_CG1_HH
#define __EQUATION_PRE2D_CG1_CG1_HH

#include <ShapeFunction.hh>
#include <LocalField.hh>
#include <GlobalField.hh>
#include <SettingsNavierStokes.hh>
#include <EquationSystem.hh>
#include <TriLinSpace.hh>
#include <math.h>

class EquationPre2d_cG1cG1:public Equation {
  
public:

  EquationPre2d_cG1cG1() : Equation(2) {
    
    start_vector_component = 2;
    
    AllocateFields(17);
    
    field[0] = &upTS[0];
    field[1] = &upTS[1];
    field[2] = &ppTS[0];
    
    field[3] = &upNL[0];
    field[4] = &upNL[1];
    field[5] = &ppNL[0];
    
    field[6]  = &unow[0];
    field[7]  = &unow[1];
    field[8] = &pnow[0];
    
    field[9] = &u_fine[0];
    field[10] = &u_fine[1];
    field[11] = &p_fine[0];
    
    field[12] = &u_coarse[0];
    field[13] = &u_coarse[1];
    field[14] = &p_coarse[0];

    field[15] = &f[0];
    field[16] = &f[1];
	 
    settings->Get("reynolds number",&Re);

    C1 = 1.0;
    C2 = 1.0;
    Cs = 0.1;

    turbulence_model = false;
    variational_multiscale_eddy_viscosity = false;
    smagorinsky_eddy_viscosity = false;
  }
  //---------------------------------------------------------------------------
  real IntegrateLHS(ShapeFunction &p, ShapeFunction &q){
    
    return ( p.dx*q.dx + p.dy*q.dy );
    
  }
  //---------------------------------------------------------------------------
  real IntegrateRHS(ShapeFunction &q){

    NL = 0.5 * ( ( (upTS[0].dx+unow[0].dx)*U[0] + 
		   (upTS[0].dy+unow[0].dy)*U[1] ) * q.dx + 
		 ( (upTS[1].dx+unow[1].dx)*U[0] + 
		   (upTS[1].dy+unow[1].dy)*U[1] ) * q.dy );
    
    DIV = 0.5 * ( upTS[0].dx + unow[0].dx +
		  upTS[1].dy + unow[1].dy ) * q;
    
    FORCE = f[0]*q.dx + f[1]*q.dy;
    
    return( FORCE - NL - (1.0/delta1)*DIV );
  }
  //---------------------------------------------------------------------------
  void UpdateRHS(){

    //display->Message(0,"Updating RHS for pressure");
	 
    // normU = sqrt( sqr(unow[0].GetMeanValue()) +
    //               sqr(unow[1].GetMeanValue()) );
    
    if (h * Re > 1.0){
      //delta1 = 0.5 / sqrt( 1.0/sqr(dt) + sqr(normU/h) );
      delta1 = C1 * h;
      delta2 = C2 * h;
    }
    else{
      delta1 = C1 * sqr(h);
      delta2 = C2 * sqr(h);
    }
    
    nu = (1.0 / Re);
    
    U[0].Mean(upTS[0],upNL[0]);
    U[1].Mean(upTS[1],upNL[1]);
    
    UpdateTurbulenceModel();
  }
  //---------------------------------------------------------------------------
  void UpdateTurbulenceModel(){
    
    if ( !turbulence_model )
      return;
    
	 S_abs = sqrt( sqr(upNL[0].dx) + 
		       sqr(0.5*(upNL[0].dy+upNL[1].dx)) + 
		       sqr(0.5*(upNL[1].dx+upNL[0].dy)) + 
		       sqr(upNL[1].dy) ); 
	 
	 if (smagorinsky_eddy_viscosity)
	   nu += pow(Cs * h, 2.0) * S_abs;;
	 
  }
  //---------------------------------------------------------------------------
  
private:

  real delta1, delta2;
  real C1, C2;

  real Re;

  real Cs, S_abs, nu, nu_T;

  real MASS, NL, NLSD, LAP, DIV, FORCE, FORCESD, PRE, PRESD;

  bool variational_multiscale_eddy_viscosity;
  bool smagorinsky_eddy_viscosity;
  bool turbulence_model;
  
  LocalField u_fine[2];
  LocalField p_fine[1];
  LocalField u_coarse[2];
  LocalField p_coarse[1];

  LocalField upTS[2];
  LocalField ppTS[1];
  LocalField upNL[2];
  LocalField ppNL[1];
  LocalField unow[2];
  LocalField pnow[1];

  LocalField f[2];
  LocalField U[2];

};

#endif
