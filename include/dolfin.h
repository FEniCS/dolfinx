#ifndef __DOLFIN_H
#define __DOLFIN_H

// Data types
typedef double real;
enum bc_type { dirichlet , neumann};

// Boundary conditions
class dolfin_bc{
public:
  dolfin_bc(){ type=neumann; val=0.0; }  
  bc_type type;
  real val;
};

// Main function calls
void dolfin_init (int argc, char **argv);
void dolfin_end  ();
void dolfin_solve();

// Specification of problem
void dolfin_set_problem(const char *problem);

// Specification of boundary conditions
void dolfin_set_boundary_conditions(dolfin_bc (*bc)(real x, real y, real z, int node, int component));
											  
// Parameters
void dolfin_set_parameter  (const char *identifier, ...);
void dolfin_get_parameter  (const char *identifier, ...);
void dolfin_save_parameters();
void dolfin_save_parameters(const char *filename);
void dolfin_load_parameters();
void dolfin_load_parameters(const char *filename);

// Functions
void dolfin_set_function(const char *identifier,
								 real (*f)(real x, real y, real z, real t));

// For testing internal functions
void dolfin_test();
void dolfin_test_memory();

#endif
