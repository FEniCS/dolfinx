// Default values for the DOLFIN parameter system.
//
// First added:  2005-12-19
// Last changed: 2007-02-27

//--- General parameters ---

add("solution file name", "solution.pvd");

//--- Parameters for input/output ---

add("save each mesh", false);

//--- Parameters for ODE solvers ---

add("ODE fixed time step", false);
add("ODE solve dual problem", false);
add("ODE save solution", true);
add("ODE save final solution", false);
add("ODE adaptive samples", false);
add("ODE automatic modeling", false);
add("ODE implicit", false);
add("ODE matrix piecewise constant", true);
add("ODE M matrix constant", false);
add("ODE monitor convergence", false);
add("ODE updated jacobian", false);        // only multi-adaptive Newton
add("ODE diagonal newton damping", false); // only multi-adaptive fixed-point
add("ODE matrix-free jacobian", true);

add("ODE order", 1);
add("ODE number of samples", 100);
add("ODE sample density", 1);
add("ODE maximum iterations", 100);
add("ODE maximum local iterations", 2);
add("ODE average samples", 1000);
add("ODE size threshold", 50);

add("ODE tolerance", 0.1);
add("ODE start time", 0.0);
add("ODE end time", 10.0);      
add("ODE discrete tolerance", 0.001);
add("ODE discrete tolerance factor", 0.001);
add("ODE discrete Krylov tolerance factor", 0.01);
add("ODE initial time step", 0.01);
add("ODE maximum time step", 0.1);
add("ODE partitioning threshold", 0.1);
add("ODE interval threshold", 0.9);
add("ODE safety factor", 0.9);
add("ODE time step conservation", 5.0);
add("ODE sparsity check increment", 0.01);
add("ODE average length", 0.1);
add("ODE average tolerance", 0.1);
add("ODE fixed-point damping", 1.0);
add("ODE fixed-point stabilize", false);
add("ODE fixed-point stabilization m", 3);
add("ODE fixed-point stabilization l", 4);
add("ODE fixed-point stabilization ramp", 2.0);

add("ODE method", "cg");
add("ODE nonlinear solver", "default");
add("ODE linear solver", "auto");
add("ODE solution file name", "solution.py");

//--- Parameters for homotopy solver ---

add("homotopy maximum size", std::numeric_limits<int>::max());
add("homotopy maximum degree", std::numeric_limits<int>::max());
add("homotopy solution tolerance", 1e-12);
add("homotopy divergence tolerance", 10.0);
add("homotopy randomize", true);
add("homotopy monitoring", false);
add("homotopy solution file name", "solution.data");

//--- Parameters for Newton solver ---

add("Newton maximum iterations", 50);
add("Newton relative tolerance", 1e-9);
add("Newton absolute tolerance", 1e-20);
add("Newton convergence criterion", "residual");
add("Newton method", "full");
add("Newton report", true);

//--- Parameters for Krylov solvers ---

add("Krylov relative tolerance", 1e-9);
add("Krylov absolute tolerance", 1e-20);
add("Krylov divergence limit",   1e4);
add("Krylov maximum iterations", 10000);
add("Krylov GMRES restart", 30);
add("Krylov shift nonzero", 0.0);
add("Krylov report", true);
add("Krylov monitor convergence", false);

//--- Parameter for direct (LU) solver ---
add("LU report", true);

//--- Parameter for PDE solver ---
add("PDE linear solver", "direct");

//--- Mesh partitioning ---
add("report edge cut", false);
