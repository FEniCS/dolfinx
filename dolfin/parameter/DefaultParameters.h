// Default values for the DOLFIN parameter system.
//
// First added:  2005-12-19
// Last changed: 2009-03-16

//--- Linear algebra ---
#ifdef HAS_PETSC
add("linear algebra backend", "PETSc");   // Use PETSc if available
#else
add("linear algebra backend", "uBLAS");   // Otherwise, use uBLAS
#endif

//--- JIT compiler ---
add("optimize form", false);              // Use optimization -O2 when compiling generated code
add("optimize use dof map cache", false); // Store dof maps in cache for reuse
add("optimize use tensor cache", false);  // Store tensors in cache for reuse
add("optimize", false);                   // All of the above

//--- General parameters ---

add("timer prefix",  "");                      // Prefix for timer tasks
add("plot file name", "dolfin_plot_data.xml"); // Name of temporary files for plot data

// FIXME: Need to cleanup among parameters below

//--- Parameters for input/output ---

add("save each mesh", false);

//--- Parameters for homotopy solver ---
add("homotopy maximum size", std::numeric_limits<int>::max());
add("homotopy maximum degree", std::numeric_limits<int>::max());
add("homotopy solution tolerance", 1e-12);
add("homotopy divergence tolerance", 10.0);
add("homotopy randomize", true);
add("homotopy monitoring", false);
add("homotopy solution file name", "solution.data");

//--- Mesh partitioning ---
add("report edge cut", false);

//--- Floating-point precision (only relevant when using GMP) ---
#ifdef HAS_GMP
add("floating-point precision", 30);
#else
add("floating-point precision", 16);
#endif
