// Copyright (C) Glenn Terje Lines, Ola Skavhaug and Simula Research Laboratory.
// Licensed under the GNU LGPL Version 2.1.
//
// Original code copied from PyCC.
// Modified by Anders Logg, 2006-2008.
//
// First added:  2006-05-24
// Last changed: 2008-10-07
//
// This demo solves the Courtemanche model for cardiac excitation.

#include <dolfin.h>

using namespace dolfin;

class Courtemanche : public ODE
{
public:
  
  Courtemanche() : ODE(21, 300.0)
  {
    // Set parameters
    Cm        = 100.0;
    R         = 8.3143; 
    T         = 310; 
    F         = 96.4867; 
    z_Na      = 1.0; 
    z_K       = 1.0; 
    z_Ca      = 2.0;
    Na_o      = 140.0; 
    K_o       = 5.4; 
    Ca_o      = 1.8; 
    K_Q10     = 3.0;
    tau_fca   = 2.0;
    k_rel     = 30.0;
    g_CaL     = 0.12375;
    gamma     = 0.35;
    I_NaCamax = 1600.0;
    K_mNa     = 87.5;
    K_mCa     = 1.38;
    k_sat     = 0.1;
    Vrel      = 96.48;
    tau_u     = 8.0;
    Vi        = 13668.0;
    I_NaKmax  = 0.59933874;
    K_mNai    = 10.0;
    //K_o=5.4;
    K_mKo     = 1.5;
    g_bNa     = 0.0006744375;
    g_Na      = 7.8;
    Vup       = 1109.52;
    I_pCamax  = 0.275;
    g_bCa     = 0.001131;
    I_upmax   = 0.005;
    K_up      = 0.00092;
    Ca_upmax  = 15.0;
    Trpn_max  = 0.070;
    K_mTrpn   = 0.0005;
    Cmdn_max  = 0.050;
    K_mCmdn   = 0.00238;
    tau_tr    = 180.0;
    Csqn_max  = 10.0;
    K_mCsqn   = 0.8;
    g_K1      = 0.09;
    g_to      = 0.1652;
    g_Kr      = 0.029411765;
    g_Ks      = 0.12941176;
    g_Na      = 7.8;
    ist       = 0.0;

    num_fevals = 0;
    VT = 0.0;
  }
  
  ~Courtemanche()
  {
    message("Function evaluations:  %d", num_fevals);
    message("Potential at end time: %.6f", VT);
  }

  void u0(double* u)
  {
    // Set initial data
    u[0]  = -85.0; 
    u[1]  = 2.91e-3; 
    u[2]  = 9.65e-1;
    u[3]  = 9.78e-1;
    u[4]  = 3.04e-2;
    u[5]  = 9.99e-1;
    u[6]  = 4.96e-3;
    u[7]  = 9.99e-1;
    u[8]  = 3.29e-5;
    u[9]  = 1.87e-2;
    u[10] = 1.37e-4;
    u[11] = 9.99e-1; 
    u[12] = 7.75e-1;
    u[13] = 0.0;
    u[14] = 1.0;
    u[15] = 9.99e-1;
    u[16] = 11.2;
    u[17] = 1.02e-4;
    u[18] = 1.49;
    u[19] = 1.49;
    u[20] = 139.0;

    // Initial kick
    u[0] = -25.0;
  }
  
  void f(const double* u, double t, double* y)
  {
    computeCurrents(u);
    computeGateCoefficients(u);

    y[0] = -1.0/Cm*(I_ion + ist);
    y[1] = (m_inf - m)/tau_m;
    y[2] = (h_inf - h)/tau_h;
    y[3] = (j_inf - j)/tau_j;
    y[4] = (oa_inf - oa)/tau_oa;
    y[5] = (oi_inf - oi)/tau_oi;
    y[6] = (ua_inf - ua)/tau_ua;
    y[7] = (ui_inf - ui)/tau_ui;
    y[8] = (xr_inf - xr)/tau_xr;
    y[9] = (xs_inf - xs)/tau_xs;
    y[10] = (d_inf - d)/tau_d;
    y[11] = (f_inf - ff)/tau_f;
    y[12] = (fca_inf - fca)/tau_fca;
    y[13] = (u_inf - uu)/tau_u;
    y[14] = (v_inf - v)/tau_v;
    y[15] = (w_inf - w)/tau_w;
    y[16] = (-3.0*I_NaK - 3.0*I_NaCa - I_bNa - I_Na)/(F*Vi);
    y[17] = B1/B2;
    y[18] = (I_tr - I_rel)/(1.0 + Csqn_max*K_mCsqn/((Ca_rel + K_mCsqn)*(Ca_rel + K_mCsqn)));
    y[19] = I_up - I_upleak - I_tr*(Vrel/Vup);
    y[20] = (2.0*I_NaK - I_K1 - I_to - I_Kur - I_Kr - I_Ks)/(F*Vi);

    num_fevals++;
  }

  void computeCurrents(const double* u)
  {
    V      = u[0];
    m      = u[1];
    h      = u[2];
    j      = u[3];
    oa     = u[4];
    oi     = u[5];
    ua     = u[6];
    ui     = u[7];
    xr     = u[8];
    xs     = u[9];
    d      = u[10];
    ff     = u[11];
    fca    = u[12];
    uu     = u[13];
    v      = u[14];
    w      = u[15];
    Na_i   = u[16];
    Ca_i   = u[17];
    Ca_rel = u[18];
    Ca_up  = u[19];
    K_i    = u[20];

    I_rel    = k_rel*uu*uu*v*w*(Ca_rel - Ca_i);
    I_CaL    = Cm*g_CaL*d*ff*fca*(V - 65);
    I_NaCa   = Cm*(I_NaCamax*(exp(gamma*F*V/(R*T))*Na_i*Na_i*Na_i*Ca_o - exp((gamma - 1)*F*V/(R*T))*Na_o*Na_o*Na_o*Ca_i))/((K_mNa*K_mNa*K_mNa + Na_o*Na_o*Na_o)*(K_mCa + Ca_o)*(1 + k_sat*exp((gamma -1 )*F*V/(R*T))));
    //Fn = 1e-12*Vrel*I_rel - 5e-13/F*(0.5*I_CaL - 0.2*I_NaCa);
    sigma    = (1.0/7.0)*(exp(Na_o/67.3) - 1.0);
    f_NaK    = 1.0/(1.0 + 0.1245*exp(-0.1*F*V/(R*T))+ 0.0365*sigma*exp(-F*V/(R*T)));
    I_NaK    = Cm*I_NaKmax*f_NaK/(1.0 + pow((K_mNai/Na_i),1.5))*(K_o/(K_o + K_mKo));
    E_Na     = R*T/(z_Na*F)*log(Na_o/Na_i);
    I_bNa    = Cm*g_bNa*(V - E_Na);
    I_Na     = Cm*g_Na*m*m*m*h*j*(V - E_Na);
    I_pCa    = Cm*I_pCamax*Ca_i/(0.0005 + Ca_i);
    E_Ca     = R*T/(z_Ca*F)*log(Ca_o/Ca_i);
    I_bCa    = Cm*g_bCa*(V - E_Ca);
    I_upleak = (Ca_up/Ca_upmax)*I_upmax;
    I_up     = I_upmax/(1.0 + (K_up/Ca_i));
    I_tr     = (Ca_up - Ca_rel)/tau_tr;
    E_K      = R*T/(z_K*F)*log(K_o/K_i);
    I_K1     = Cm*(g_K1*(V - E_K))/(1.0 + exp(0.07*(V + 80.0)));
    I_to     = Cm*g_to*oa*oa*oa*oi*(V - E_K);
    g_Kur    = 0.005 + 0.05/(1.0 + exp((V - 15.0)/-13.0));
    I_Kur    = Cm*g_Kur*ua*ua*ua*ui*(V - E_K);
    I_Kr     = Cm*(g_Kr*xr*(V - E_K))/(1.0 + exp((V + 15.0)/22.4));
    I_Ks     = Cm*g_Ks*xs*xs*(V - E_K);
    I_ion    = I_Na + I_K1 + I_to + I_Kur + I_Kr + I_Ks + I_CaL + I_pCa + I_NaK + I_NaCa + I_bNa + I_bCa;
    B1       = (2.0*I_NaCa - I_pCa - I_CaL - I_bCa)/(2.0*F*Vi) + (Vup*(I_upleak - I_up) + I_rel*Vrel)/Vi;
    B2       = 1.0 + Trpn_max*K_mTrpn/((Ca_i + K_mTrpn)*(Ca_i + K_mTrpn)) + Cmdn_max*K_mCmdn/((Ca_i + K_mCmdn)*(Ca_i + K_mCmdn));
  }
  
  void computeGateCoefficients(const double* u)
  {
    V = u[0];
    
    if ( V == -47.13 )
        alpha_m = 3.2;
    else
        alpha_m = 0.32*(V + 47.13)/(1.0 - exp(-0.1*(V + 47.13)));

    beta_m = 0.08*exp(V/-11.0);
    tau_m  = 1.0/(alpha_m + beta_m);
    m_inf  = alpha_m*tau_m;
    if (V >= -40.0){
        alpha_h = 0.0;
        beta_h  = 1.0/(0.13*(1.0 + exp((V + 10.66)/-11.1)));
    } else {
        alpha_h = 0.135*exp((V + 80.0)/-6.8);
        beta_h  = 3.56*exp(0.079*V)+3.1e5*exp(0.35*V);
    }

    tau_h = 1.0/(alpha_h + beta_h);
    h_inf = alpha_h*tau_h;
      
    if ( V >= -40.0 )
    {
        alpha_j = 0.0;
        beta_j  = 0.3*(exp(-2.535e-7*V))/(1.0 + exp(-0.1*(V +32.0)));
    } else {
        alpha_j = (-127140.0*exp(0.2444*V)-3.474e-5*exp(-0.04391*V))*(V + 37.78)/(1.0 + exp(0.311*(V + 79.23)));
        beta_j  = 0.1212*(exp(-0.01052*V))/(1.0 + exp(-0.1378*(V + 40.14)));
    }

    tau_j = 1.0/(alpha_j + beta_j);
    j_inf = alpha_j*tau_j;
      
    alpha_oa = 0.65/(exp((V + 10.0)/-8.5) + exp((V - 30.0)/-59.0));
    beta_oa  = 0.65/(2.5 + exp((V + 82.0)/17.0));
    tau_oa   = 1.0/((alpha_oa + beta_oa)*K_Q10);
    oa_inf   = 1.0/(1.0 + exp((V + 20.47)/-17.54));
      
    alpha_oi = 1.0/(18.53 + exp((V + 113.7)/10.95));
    beta_oi  = 1.0/(35.56 + exp((V + 1.26)/-7.44));
    tau_oi   = 1.0/((alpha_oi + beta_oi)*K_Q10);
    oi_inf   = 1.0/(1.0 + exp((V + 43.1)/5.3));
      
    alpha_ua = 0.65/(exp((V + 10.0)/-8.5) + exp((V - 30)/-59.0));
    beta_ua  = 0.65/(2.5 + exp((V + 82.0)/17.0));
    tau_ua   = 1.0/((alpha_ua + beta_ua)*K_Q10);
    ua_inf   = 1.0/(1.0 + exp((V + 30.3)/-9.6));
     
    alpha_ui = 1.0/(21.0 + exp((V - 185.0)/-28.0));
    beta_ui  = exp((V - 158.0)/16.0);
    tau_ui   = 1.0/((alpha_ui + beta_ui)*K_Q10);
    ui_inf   = 1.0/(1.0 + exp((V - 99.45)/27.48));
      
    alpha_xr = 0.0003*(V + 14.1)/(1.0 - exp((V + 14.1)/-5.0));
    beta_xr  = 7.3898e-05*(V - 3.3328)/(exp((V -3.3328)/5.1237) - 1.0);
    tau_xr   = 1.0/(alpha_xr + beta_xr);
    xr_inf   = 1.0/(1.0 + exp((V + 14.1)/-6.5));
      
    alpha_xs = 4e-05*(V - 19.9)/(1.0 - exp((V - 19.9)/-17.0));
    beta_xs  = 3.5e-05*(V - 19.9)/(exp((V - 19.9)/9.0) - 1.0);
    tau_xs   = 0.5/(alpha_xs + beta_xs);
    xs_inf   = pow((1.0 + exp((V - 19.9)/-12.7)),-0.5);
     
    tau_d    = (1.0 - exp((V + 10.0)/-6.24))/(0.035*(V + 10.0)*(1.0 + exp((V + 10.0)/-6.24))); 
    d_inf    = 1.0/(1.0 + exp((V +10.0)/-8.0));
      
    tau_f    = 9.0/(0.0197*exp(-0.0337*0.0337*(V + 10.0)*(V + 10.0)) + 0.02); 
    f_inf    = 1.0/(1.0 + exp((V + 28.0)/6.9));
     
    fca_inf  = 1.0/(1.0 + Ca_i/0.00035);
     
    Fn       = 1e-12*Vrel*I_rel - (5e-13/F)*(0.5*I_CaL - 0.2*I_NaCa);
    u_inf    = 1.0/(1.0 + exp((Fn - 3.4175e-13)/-13.67e-16));
     
    tau_v    = 1.91 + 2.09/(1.0 + exp((Fn - 3.4175e-13)/-13.67e-16));
    v_inf    = 1.0 - 1.0/(1.0 + exp((Fn - 6.835e-14)/-13.67e-16));
     
    tau_w    = 6.0*(1.0 - exp((V - 7.9)/-5.0))/((1.0 + 0.3*exp((V - 7.9)/-5.0))*(V - 7.9));
    w_inf    = 1.0 - 1.0/(1.0 + exp((V - 40.0)/-17.0));
  }
  
  bool update(const double* u, double t, bool end)
  {
    if ( end )
      VT = u[0];
    return true;
  }
  
private:
  
  // State varibles
  double m, h, j, oa, oi, ua, ui, xr, xs, d, ff, fca, uu, v, w, Na_i, Ca_i;
  double Ca_rel, Ca_up, K_i, V;
  
  // Ionic currents and gating variables
  double alpha_m, beta_m, tau_m, m_inf, alpha_h, beta_h, tau_h, h_inf;
  double alpha_j, beta_j, tau_j, j_inf, alpha_oa, beta_oa, tau_oa, oa_inf;
  double alpha_oi, beta_oi, tau_oi, oi_inf, alpha_ua, beta_ua, tau_ua, ua_inf;
  double alpha_ui, beta_ui, tau_ui, ui_inf, alpha_xr, beta_xr, tau_xr, xr_inf;
  double alpha_xs, beta_xs, tau_xs, xs_inf, tau_d, d_inf, tau_f, f_inf;
  double fca_inf, u_inf, tau_v, v_inf, tau_w, w_inf, B1, B2;
  
  // Membrane currents
  double I_rel, I_CaL, I_NaCa, Fn, sigma, f_NaK, I_NaK, E_Na, I_bNa;
  double I_Na, I_pCa, E_Ca, I_bCa, I_upleak, I_up, I_tr, E_K, I_K1;
  double I_to, g_Kur, I_Kur, I_Kr, I_Ks, I_ion;
  
  // Gate coefficients
  double Cm, R, T, F, z_Na, z_K, z_Ca, Na_o, K_o, Ca_o, K_Q10, tau_fca;
  double k_rel, g_CaL, gamma, I_NaCamax, K_mNa, K_mCa, k_sat, Vrel, tau_u;
  double Vi, I_NaKmax, K_mNai, K_mKo, g_bNa, g_Na, Vup, I_pCamax, g_bCa;
  double I_upmax, K_up, Ca_upmax, Trpn_max, K_mTrpn, Cmdn_max, K_mCmdn;
  double tau_tr, Csqn_max, K_mCsqn, g_K1, g_to, g_Kr, g_Ks;
  double Na_e, Ca_e, K_e;

  // Stimulus current
  double ist;
  
  // Number of function evaluations
  unsigned int num_fevals;

  // Value at end time
  double VT;

};

int main()
{
  dolfin_set("ODE tolerance", 1.0e-5);
  dolfin_set("ODE maximum time step", 100.0);
  dolfin_set("ODE nonlinear solver", "newton");
  dolfin_set("ODE linear solver", "iterative");
  dolfin_set("ODE initial time step", 0.25);

  //dolfin_set("ODE save solution", false);

  Courtemanche ode;
  ode.solve();

  return 0;
}
