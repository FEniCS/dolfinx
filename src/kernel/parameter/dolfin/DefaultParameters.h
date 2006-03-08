// Default values for the DOLFIN parameter system.
//
// First added:  2005-12-19
// Last changed: 2005-12-19

//--- General parameters ---

add("solution file name", "solution.pvd");

//--- Parameters for input/output ---

add("save each mesh", false);

//--- Parameters for ODE solvers ---

add("fixed time step", false);
add("solve dual problem", false);
add("save solution", true);
add("save final solution", false);
add("adaptive samples", false);
add("automatic modeling", false);
add("implicit", false);
add("matrix piecewise constant", true);
add("monitor convergence", false);
add("updated jacobian", false);        // only multi-adaptive Newton
add("diagonal newton damping", false); // only multi-adaptive fixed-point

add("order", 1);
add("number of samples", 101);
add("sample density", 1);
add("maximum iterations", 100);
add("maximum local iterations", 2);
add("average samples", 1000);

add("tolerance", 0.1);
add("start time", 0.0);
add("end time", 10.0);      
add("discrete tolerance", 0.001);
add("discrete tolerance factor", 0.001);
add("initial time step", 0.01);
add("maximum time step", 0.1);
add("partitioning threshold", 0.1);
add("interval threshold", 0.9);
add("safety factor", 0.9);
add("time step conservation", 5.0);
add("sparsity check increment", 0.01);
add("average length", 0.1);
add("average tolerance", 0.1);

add("method", "cg");
add("solver", "default");
add("linear solver", "default");
add("ode solution file name", "primal.m");

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
