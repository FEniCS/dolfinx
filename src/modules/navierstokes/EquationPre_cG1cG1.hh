#ifndef __EQUATION_PRE_CG1_CG1_HH
#define __EQUATION_PRE_CG1_CG1_HH

#include <ShapeFunction.hh>
#include <LocalField.hh>
#include <GlobalField.hh>
#include <SettingsNavierStokes.hh>
#include <EquationSystem.hh>
#include <TetLinSpace.hh>
#include <math.h>

class EquationPre_cG1cG1:public Equation {
  
public:

  EquationPre_cG1cG1() : Equation(3) {
    
    start_vector_component = 3;
    
    AllocateFields(23);
    
    field[0] = &upTS[0];
    field[1] = &upTS[1];
    field[2] = &upTS[2];
    field[3] = &ppTS[0];
    
    field[4] = &upNL[0];
    field[5] = &upNL[1];
    field[6] = &upNL[2];
    field[7] = &ppNL[0];
    
    field[8]  = &unow[0];
    field[9]  = &unow[1];
    field[10] = &unow[2];
    field[11] = &pnow[0];
    
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
    
    return ( p.dx*q.dx + p.dy*q.dy + p.dz*q.dz );
    
  }
  //---------------------------------------------------------------------------
  real IntegrateRHS(ShapeFunction &q){

    NL = 0.5 * ( ( (upTS[0].dx+unow[0].dx)*U[0] + 
						 (upTS[0].dy+unow[0].dy)*U[1] + 
						 (upTS[0].dz+unow[0].dz)*U[2] ) * q.dx + 
					  ( (upTS[1].dx+unow[1].dx)*U[0] + 
						 (upTS[1].dy+unow[1].dy)*U[1] + 
						 (upTS[1].dz+unow[1].dz)*U[2] ) * q.dy + 
					  ( (upTS[2].dx+unow[2].dx)*U[0] + 
						 (upTS[2].dy+unow[2].dy)*U[1] + 
						 (upTS[2].dz+unow[2].dz)*U[2] ) * q.dz );
    
    DIV = 0.5 * ( upTS[0].dx + unow[0].dx +
						upTS[1].dy + unow[1].dy +
						upTS[2].dz + unow[2].dz ) * q;
    
    FORCE = f[0]*q.dx + f[1]*q.dy + f[2]*q.dz;
    
    return( FORCE - NL - (1.0/delta1)*DIV );
  }
  //---------------------------------------------------------------------------
  void UpdateRHS(){

    // normU = sqrt( sqr(unow[0].GetMeanValue()) +
    //               sqr(unow[1].GetMeanValue()) +
    //               sqr(unow[2].GetMeanValue())) ;
    
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
    U[2].Mean(upTS[2],upNL[2]);
    
    UpdateTurbulenceModel();
  }
  //---------------------------------------------------------------------------
  void UpdateTurbulenceModel(){
    
    if ( !turbulence_model )
      return;
    
	 S_abs = sqrt( sqr(upNL[0].dx) + 
		       sqr(0.5*(upNL[0].dy+upNL[1].dx)) + 
		       sqr(0.5*(upNL[0].dz+upNL[2].dx)) + 
		       sqr(0.5*(upNL[1].dx+upNL[0].dy)) + 
		       sqr(upNL[1].dy) + 
		       sqr(0.5*(upNL[1].dz+upNL[2].dy)) + 
		       sqr(0.5*(upNL[2].dx+upNL[0].dz)) + 
		       sqr(0.5*(upNL[2].dy+upNL[1].dz)) + 
		       sqr(upNL[2].dz) );
	 
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
  
  LocalField u_fine[3];
  LocalField p_fine[1];
  LocalField u_coarse[3];
  LocalField p_coarse[1];

  LocalField upTS[3];
  LocalField ppTS[1];
  LocalField upNL[3];
  LocalField ppNL[1];
  LocalField unow[3];
  LocalField pnow[1];

  LocalField f[3];
  LocalField U[3];

};

#endif
