// Copyright (C) 2009 Mirko Maraldi and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-01-22
// Last changed: 2009-01-27
//

#include <dolfin.h>
#include "ElastoDynamics.h"
#include "DG0_eps_xx.h"

using namespace dolfin;

// External load
class Pressure : public Function
{
  public:
    Pressure(const double &t, const double &dt, bool old): t(t), dt(dt), old(old) {}

  private:
  	void eval(double* values, const Data& data) const
  	{
      double time = t;
      if(old && time > 0.0)
        time -= dt;

      const double cutoff_time = 10.0*1.0/32.0;
      if (time < cutoff_time)
      {
  		  values[0] = 1.0*time/cutoff_time;
  		  values[1] = 0.0;
      }
      else
      {
  		  values[0] = 1.0;
  		  values[1] = 0.0;
      }
  	}
    const double& t;
    const double& dt;
    const bool    old;
};


// Right boundary
class RightBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    if( 1.0-x[0] < DOLFIN_EPS && on_boundary)
      return true;
    else
      return false;
  }
};

class LeftBoundary : public SubDomain
{
  bool inside(const double* x, bool on_boundary) const
  {
    if( x[0] < DOLFIN_EPS )
      return true;
    else
      return false;
  }
};

// Acceleration update
void update_a(Function& a, const Function& u, const Function& a0, 
              const Function& v0,  const Function& u0, 
              double Beta, double delta_t)
{
  // a = 1/(2*Beta) * ((u-u0 - v0*dt)/(0.5*dt*dt) - (1-2*Beta)*a0)
  a.vector()  = u.vector(); 
  a.vector() -= u0.vector(); 
  a.vector() *= 1.0/delta_t;
  a.vector() -= v0.vector();
  a.vector() *= 1.0/((0.5-Beta)*delta_t);
  a.vector() -= a0.vector();
  a.vector() *= (0.5-Beta)/Beta;
}

// Velocity update
void update_v(Function& v, const Function& a, const Function& a0, 
              const Function& v0, double Gamma, double delta_t)
{
  // v = dt * ((1-Gamma)*a0 + Gamma*a) + v0
  v.vector()  = a0.vector();
  v.vector() *= (1.0-Gamma)/Gamma;
  v.vector() += a.vector();
  v.vector() *= delta_t*Gamma;
  v.vector() += v0.vector();
}

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // Create Mesh
  Mesh mesh("../../../../data/meshes/dolfin-2.xml.gz");

  // Create function space
  ElastoDynamicsFunctionSpace V(mesh);

	// Material parameters
  Constant rho(1.0);                           // mass density
  Constant eta(0.25);                          // damping coefficient
  double E  = 1.0;                             // Youngs modulus
  double nu = 0.0;                             // Poisson ratio 
  Constant lambda((nu*E)/((1.0+nu)*(1.0-nu))); // Lame coefficient
  Constant mu(E/(2.0*(1.0+nu)));               // Lame coefficient

  // Time stepping parameters
  Constant alpha_m(0.2);
  Constant alpha_f(0.4);
  double Beta = 0.36;                   
  double Gamma = 0.7;      
  double delta_t = 1.0/32.0;               // time step
  double t = 0.0;                              // initial time
  double T = 200*delta_t;                      // final time
  Constant beta(Beta), gamma(Gamma), dt(delta_t);

  // Body force
  Constant f(2, 0.0);

  // External load
  RightBoundary right_boundary;
  MeshFunction<unsigned int> right_boundary_function(mesh, 1);
  right_boundary.mark(right_boundary_function, 3);
  Pressure p(t, delta_t, false), p0(t, delta_t, true);
  
  // Dirichlet boundary conditions
  LeftBoundary left_boundary;
  Constant zero(2, 0.0);
  DirichletBC bc0(V, zero, left_boundary);
  std::vector<BoundaryCondition*> bc;
  bc.push_back(&bc0);

  // Define solution vectors
  Function u(V), u0(V);           // displacement
  Function v(V), v0(V);           // velocity
  Function a(V), a0(V);           // acceleration

  // Set initial conditions and initialise acceleration function
  u0.vector().zero();
  v0.vector().zero();
  a0.vector().zero();

  // Create forms and attach functions
  ElastoDynamicsBilinearForm a_form(V, V);
  a_form.lmbda = lambda; a_form.mu = mu; a_form.rho = rho; 
  a_form.eta = eta;
  a_form.beta = beta; 
  a_form.gamma = gamma;
  a_form.dt = dt; 
  a_form.alpha_m = alpha_m; 
  a_form.alpha_f = alpha_f;

  ElastoDynamicsLinearForm L(V);
  L.u0 = u0; L.v0 = v0; L.a0 = a0;
  L.lmbda = lambda; 
  L.mu = mu; 
  L.rho = rho; 
  L.eta = eta;
  L.beta = beta; 
  L.gamma = gamma;
  L.dt = dt; 
  L.alpha_m = alpha_m; 
  L.alpha_f = alpha_f;
  L.f = f; 
  L.p = p; 
  L.p0 = p0;

  // Create variational problem
  VariationalProblem pde(a_form, L, bc, 0, &right_boundary_function, 0);

  // Create projection to compute the normal strain eps_xx
  DG0_eps_xxFunctionSpace Vdg(mesh);
  DG0_eps_xxBilinearForm a_eps(Vdg, Vdg);
  DG0_eps_xxLinearForm L_eps(Vdg);
  L_eps.u = u;
  VariationalProblem eps(a_eps, L_eps);
  Function eps_xx(Vdg);

  // Create output files
	File file_u("u.pvd");
	File file_eps("eps_xx.pvd");

  // Start time stepping
  dolfin::uint step = 0;
  while( t < T)
  {
    // Update for next time step
    t += delta_t;
    cout << "Time: " << t << endl;

    // Solve
    pde.solve(u);
    eps.solve(eps_xx);

    // Update velocity and acceleration
    update_a(a, u, a0, v0, u0, Beta, delta_t);
    update_v(v, a, a0, v0, Gamma, delta_t);
    u0 = u; v0 = v; a0 = a;

    // Save solutions to file
    if( step % 2 == 0)
    {
      file_u << u;
      file_eps << eps_xx;
    }
    ++step;
	}
  
  return 0;
}
