// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.

#ifndef __EULER_H
#define __EULER_H

#include <dolfin/PDE.h>

namespace dolfin 
{

  class Euler : public PDE 
  {
  
  public:

    Euler(Function::Vector& SourceMomentum,
	  Function SourceEnergy,
	  Function FluidViscosity,
	  Function FluidConductivity,
	  Function::Vector& ulinear, 
	  Function::Vector& uprevious) :
      PDE(3,5), fm(3), ulin(5), up(5)
    {
      add(fm,    SourceMomentum);
      add(fe,    SourceEnergy);
      add(am,    FluidViscosity);
      add(ae,    FluidConductivity);
      add(ulin,  ulinear);
      add(up,    uprevious);
    
    }

    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {
             
       ElementFunction Aij_Time_Continuity = 0.1;
       ElementFunction Aij_Conv_Continuity = 0.1;
       ElementFunction Aij_Pres_Continuity = 0.1;
       ElementFunction Aij_Diff_Continuity = 0.1;
       ElementFunction Aij_Dill_Continuity = 0.1;
       ElementFunction Aij_Stab_Continuity = 0.1;

       ElementFunction Aij_Time_Momentum = 0.01;
       ElementFunction Aij_Conv_Momentum = 0.01;
       ElementFunction Aij_Pres_Momentum = 0.01;
       ElementFunction Aij_Diff_Momentum = 0.01;
       ElementFunction Aij_Dill_Momentum = 0.01;
       ElementFunction Aij_Stab_Momentum = 0.01;

       ElementFunction Aij_Time_Energy = 0.02;
       ElementFunction Aij_Conv_Energy = 0.02;
       ElementFunction Aij_Pres_Energy = 0.02;
       ElementFunction Aij_Diff_Energy = 0.02;
       ElementFunction Aij_Dill_Energy = 0.02;
       ElementFunction Aij_Stab_Energy = 0.02;

       ElementFunction Aij_Time = 0.0;
       ElementFunction Aij_Conv = 0.0;
       ElementFunction Aij_Pres = 0.0;
       ElementFunction Aij_Diff = 0.0;
       ElementFunction Aij_Dill = 0.0;
       ElementFunction Aij_Stab = 0.0;


       Aij_Time = Aij_Time_Continuity + Aij_Time_Momentum + Aij_Time_Energy;

       Aij_Conv = Aij_Conv_Continuity + Aij_Conv_Momentum + Aij_Conv_Energy;

                   
       Aij_Pres = Aij_Pres_Continuity + Aij_Pres_Momentum + Aij_Pres_Energy;

       Aij_Diff = Aij_Diff_Continuity + am*Aij_Diff_Momentum + ae*Aij_Diff_Energy;

      return ( ( Aij_Time +  k*(Aij_Conv + Aij_Pres + Aij_Diff) ) *dx );
    }


    real rhs(ShapeFunction::Vector& v)
    {

      ElementFunction bj_Time_Continuity = 0.01;
      ElementFunction bj_Conv_Continuity = 0.02;
      ElementFunction bj_Diff_Continuity = 0.03;
      ElementFunction bj_Dill_Continuity = 0.04;
      ElementFunction bj_Stab_Continuity = 0.05;
      ElementFunction bj_Load_Continuity = 0.06;

      ElementFunction bj_Time_Momentum = 0.02;
      ElementFunction bj_Conv_Momentum = 0.03;
      ElementFunction bj_Diff_Momentum = 0.04;
      ElementFunction bj_Dill_Momentum = 0.05;
      ElementFunction bj_Stab_Momentum = 0.06;
      ElementFunction bj_Load_Momentum = 0.07;

      ElementFunction bj_Time_Energy = 0.03;
      ElementFunction bj_Conv_Energy = 0.04;
      ElementFunction bj_Diff_Energy = 0.05;
      ElementFunction bj_Dill_Energy = 0.06;
      ElementFunction bj_Stab_Energy = 0.07;
      ElementFunction bj_Load_Energy = 0.08;

      ElementFunction bj_Time = 0.0;
      ElementFunction bj_Conv = 0.0;
      ElementFunction bj_Diff = 0.0;
      ElementFunction bj_Dill = 0.0;
      ElementFunction bj_Stab = 0.0;
      ElementFunction bj_Load = 0.0;


      bj_Time = bj_Time_Continuity + bj_Time_Momentum + bj_Time_Energy;

      bj_Load = bj_Load_Continuity + bj_Load_Momentum + bj_Load_Energy;

      return ( ( bj_Time +  k*(bj_Conv + bj_Diff + bj_Load) )*dx );    
    }

  private:

    ElementFunction::Vector fm;   // Momentum Source term
    ElementFunction fe;           // Energy Source term
    ElementFunction am;           // Fluid Viscosity ( momentum diffusivity )
    ElementFunction ae;           // Fluid Conductivity ( energy diffusivity )

    ElementFunction::Vector ulin; // u linear

    ElementFunction::Vector up;   // u prevoius

  };

}

#endif
