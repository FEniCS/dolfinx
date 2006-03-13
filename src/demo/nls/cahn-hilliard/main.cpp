// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-03-02
// Last changed: 2006-03-02
//
// This program illustrates the use of the DOLFIN nonlinear solver for solving 
// the Cahn-Hilliard equation.
//
// The Cahn-Hilliard equation is very sensitive to the chosen parameters and
// time step. It also requires fines meshes, and is often not well-suited to
// iterative linear solvers.
//
// To see some interesting results, for a unit mesh, some recommned parameters are provided
//   For quartic chemical free-energy:
//     lambda = 5.0e02
//     factor = 100
//     dt     = 1.0e-6 
//
//   For logarithmic chemical free-energy:
//     lambda = 1.0
//     g1 = 0, g2 = 0, g3 = 1000, g4 = 1000, g5 = 3000
//     dt = 2.0e-7 
//

#include <dolfin.h>
#include "CahnHilliard2D.h"
#include "CahnHilliard3D.h"

using namespace dolfin;

// User defined nonlinear problem 
class CahnHilliardEquation : public NonlinearProblem, public Parametrized
{
  public:

    // Constructor 
    CahnHilliardEquation(Mesh& mesh, Function& U, Function& rate1, Function& U0, 
        Function& rate0, real& theta, real& dt, real& theta_inv, real& dt_inv) 
        : NonlinearProblem(), Parametrized(), _mesh(&mesh), U0(&U0), _dt(&dt), 
        _theta(&theta), _dt_inv(&dt_inv), _theta_inv(&theta_inv)
    {

      lambda = 1.0;
      add("mobility", "nonlinear");
      add("chemical potential", "logarithmic");


      // Create functions
      mobility = new Function; d_mobility = new Function;
      mu = new Function; dmu_dc = new Function;

      // Create forms
      if(mesh.numSpaceDim() == 2)
      {
        a = new CahnHilliard2D::BilinearForm(U, *mu, *dmu_dc, *mobility, *d_mobility, lambda, *_dt, *_theta);
        L = new CahnHilliard2D::LinearForm(U, U0, rate0, *mu, *mobility, lambda, *_dt, *_theta, *_dt_inv, 
                                           *_theta_inv);
      }
      else if(mesh.numSpaceDim() == 3)
      {
        a = new CahnHilliard3D::BilinearForm(U, *mu, *dmu_dc, *mobility, *d_mobility, lambda, *_dt, *_theta);
        L = new CahnHilliard3D::LinearForm(U, U0, rate0, *mu, *mobility, lambda, *_dt, *_theta, *_dt_inv, 
                                           *_theta_inv);
      }
      else
        dolfin_error("Cahn-Hilliard model is programmed for 2D and 3D only");

      // Initialise solution functions
      U.init(mesh, a->trial());
      rate1.init(mesh, a->trial());
      U0.init(mesh, a->trial());
      rate0.init(mesh, a->trial());

      // Initialise chemical potential and mobility functions
      mu->init(mesh, a->trial()[0]);
      dmu_dc->init(mesh, a->trial()[0]);
      mobility->init(mesh, a->trial()[0]);
      d_mobility->init(mesh, a->trial()[0]);
    }

    // Destructor 
    ~CahnHilliardEquation()
    {
      delete mobility, d_mobility;
      delete mu, dmu_dc;
      delete a, L;
    }
 
    // Compute chemical potential mu = df/dc (f =chemical free-energy)
    void computeMu(Function& mu, Function& dmu, const Vector& x)
    {
      // parameters for the logarithmic free energy
      real g1 = 0.0; real g2 = 0.0; real g3 = 1000.0; real g4 = 1000.0; real g5 = 3000.0;
      // factor on quartic free energy
      real factor = 100.0;

      // Get vectors
      Vector& mu_vector  = mu.vector();
      Vector& dmu_vector = dmu.vector();

      // Get pointer to relavnt ararys
      real* mu_array  = mu_vector.array();
      real* dmu_array = dmu_vector.array();
      const real* x_array = x.array();

      dolfin::uint m = mu_vector.size();
      real c;

      const std::string type = get("chemical potential");
      if(type == "logarithmic")
      {
        for(dolfin::uint j=0; j < m; ++j)
        {
          c = *(x_array+j);
          if( c < 0 || c > 1)
            dolfin_error("Concentration outside of allowable bounds");
          *(mu_array+j)  =  g3/c + g4/(1-c) - 2*g5;
          *(dmu_array+j) = -g3/(c*c)  + g4/( (1-c)*(1-c) );
        }
      }
      else if(type == "quartic")  // f(c) = c^2*(1-c)^2
      {
        for(dolfin::uint j=0; j < m; ++j)
        {
          c = *(x_array+j);
          *(mu_array+j)  = factor*( 2*(1-c)*(1-c) - 8*c*(1-c) + 2*c*c);
          *(dmu_array+j) = factor*(-12 + 24*c);
        }
      }
      else
        dolfin_error("Unknown chemical potential type");

      // Finalise
      mu_vector.restore(mu_array);
      dmu_vector.restore(dmu_array);
      x.restore(x_array);
    }

    // Compute mobility
    void computeMobility(Function& mob, Function& dmob, const Vector& x)
    {
      // Get vectors
      Vector& m_vector  = mob.vector();
      Vector& dm_vector = dmob.vector();

      const std::string type = get("mobility");

      if( type == "nonlinear") // M = c(1-c)
      {
        // Get pointer to ararys
        real* m_array  = m_vector.array(); 
        real* dm_array = dm_vector.array();
        const real* x_array = x.array();

        real c;
        dolfin::uint m = m_vector.size();
        for(dolfin::uint j=0; j < m; ++j)
        {
          c = *(x_array+j);
          *(m_array+j)  = c*(1-c);
          *(dm_array+j) = 1.0 - 2.0*c;
        }  
        // Finalise
        m_vector.restore(m_array);
        dm_vector.restore(dm_array);
        x.restore(x_array);
      }
      else if( type == "constant") // Constant mobility M = 1
      {
        m_vector  = 1.0;
        dm_vector = 0.0;
      }
      else
        dolfin_error("Unknown mobility type");
    }

    // User defined assemble of Jacobian and residual vector 
    void form(Matrix& A, Vector& b, const Vector& x)
    {
      // Compute chemical potential
      computeMu(*mu, *dmu_dc, x);

      // Compute Mobility
      computeMobility(*mobility, *d_mobility, x);

      // Assemble system and RHS (Neumann boundary conditions)
      dolfin_log(false);
      FEM::assemble(*a, *L, A, b, *_mesh);
      dolfin_log(true);
    }

  private:

    // Pointers to forms and  mesh
    BilinearForm *a;
    LinearForm *L;
    Mesh* _mesh;

    // Mobility and its derivative 
    Function *mobility, *d_mobility;
    // Chemical potential and its derivative 
    Function *mu, *dmu_dc;
    // Solutions and rates 
    Function *U0;
    real *_dt, *_theta, *_dt_inv, *_theta_inv;
    // Surface parameter
    real lambda;
};


int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // Set up problem
  UnitSquare mesh(40, 40);

  // Set time stepping parameters
  real dt = 2.0e-7; real t  = 0.0; real T  = 500*dt;
  real theta = 0.5;
  real theta_inv = 1.0/theta;
  real dt_inv = 1.0/dt;

  Function U, U0;
  Function rate1, rate0;

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, U, rate1, U0, rate0, theta, dt, 
                       theta_inv, dt_inv);

  // Create nonlinear solver and set parameters
  NewtonSolver newton_solver;
  newton_solver.set("Newton convergence criterion", "incremental");
  newton_solver.set("Newton maximum iterations", 10);
  newton_solver.set("Newton relative tolerance", 1e-6);
  newton_solver.set("Newton absolute tolerance", 1e-15);

  Vector& x = U.vector();

  // Randomly perturbed intitial conditions
  dolfin::seed(2);
  for(dolfin::uint i=0; i < mesh.numVertices(); ++i)
     x(i) = 0.63 + 0.02*(0.5-dolfin::rand());

  // Save initial condition to file
  File file("cahn_hilliard.pvd");
  Function c = U[0];
  file << c;

  Vector& r  = rate1.vector();
  Vector& x0 = U0.vector();
  Vector& r0 = rate0.vector();

  while( t < T)
  {
    // Update for next time step
    t += dt;
    U0    = U;
    rate0 = rate1;

    // Solve
    newton_solver.solve(cahn_hilliard, x);

    // Compute rate
    r = r0;
    r *= -((1.0-theta)/theta);
    r.axpy((1.0/(theta*dt)), x);
    r.axpy(-(1.0/(theta*dt)), x0);

   // Save function to file
    c = U[0];
    file << c;
  }

  return 0;
}
