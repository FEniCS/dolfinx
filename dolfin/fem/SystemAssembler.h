// Copyright (C) 2008-2009 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008-2009.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2009-06-22
// Last changed: 2009-06-22

#ifndef __SYSTEM_ASSEMBLER_H
#define __SYSTEM_ASSEMBLER_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class DirichletBC;
  class GenericMatrix;
  class GenericTensor;
  class GenericVector;
  class Form;
  class Mesh;
  class SubDomain;
  class UFC;
  class Cell; 
  class Facet; 
  class Function;
  template<class T> class MeshFunction;
  
  /// This class provides implements an assembler for systems
  /// of the form Ax = b. It differs from the default DOLFIN
  /// assembler in that it assembles both A and b and the same
  /// time (leading to better performance) and in that it applies
  /// boundary conditions at the time of assembly.

  class SystemAssembler
  {
  public:

    /// Assemble system (A, b)
    static void assemble_system(GenericMatrix& A,
                                GenericVector& b,
                                const Form& a,
                                const Form& L,
                                bool reset_tensors=true);

    /// Assemble system (A, b) and apply Dirichlet boundary condition
    static void assemble_system(GenericMatrix& A,
                                GenericVector& b,
                                const Form& a,
                                const Form& L,
                                const DirichletBC& bc,
                                bool reset_tensors=true);

    /// Assemble system (A, b) and apply Dirichlet boundary conditions
    static void assemble_system(GenericMatrix& A,
                                GenericVector& b,
                                const Form& a,
                                const Form& L,
                                std::vector<const DirichletBC*>& bcs,
                                bool reset_tensors=true);

    /// Assemble system (A, b) on sub domains and apply Dirichlet boundary conditions
    static void assemble_system(GenericMatrix& A,
                                GenericVector& b,
                                const Form& a,
                                const Form& L,
                                std::vector<const DirichletBC*>& bcs,
                                const MeshFunction<uint>* cell_domains,
                                const MeshFunction<uint>* exterior_facet_domains,
                                const MeshFunction<uint>* interior_facet_domains,
                                const GenericVector* x0,
                                bool reset_tensors=true);

    /// Assemble system (A, b)
    static void assemble_system_new(GenericMatrix& A,
                                GenericVector& b,
                                const Form& a,
                                const Form& L,
                                bool reset_tensors=true);

    /// Assemble system (A, b) and apply Dirichlet boundary condition
    static void assemble_system_new(GenericMatrix& A,
                                    GenericVector& b,
                                    const Form& a,
                                    const Form& L,
                                    const DirichletBC& bc,
                                    bool reset_tensors=true);

    /// Assemble system (A, b) and apply Dirichlet boundary conditions
    static void assemble_system_new(GenericMatrix& A,
                                    GenericVector& b,
                                    const Form& a,
                                    const Form& L, 
                                    std::vector<const DirichletBC*>& bcs,
                                    bool reset_tensors=true);

    /// Assemble system (A, b) and apply Dirichlet boundary conditions
    static void assemble_system_new(GenericMatrix& A,
                                    GenericVector& b,
                                    const Form& a,
                                    const Form& L,
                                    std::vector<const DirichletBC*>& bcs,
                                    const MeshFunction<uint>* cell_domains,
                                    const MeshFunction<uint>* exterior_facet_domains,
                                    const MeshFunction<uint>* interior_facet_domains,
                                    const GenericVector* x0,
                                    bool reset_tensors=true);

  private:

    static void compute_tensor_on_one_cell(const Form& a,
                                           UFC& ufc, 
                                           const Cell& cell, 
                                           const std::vector<const Function*>& coefficients, 
                                           const MeshFunction<uint>* cell_domains); 
    
    static void compute_tensor_on_one_exterior_facet(const Form& a,
                                               UFC& ufc, 
                                               const Cell& cell, 
                                               const Facet& facet,
                                               const std::vector<const Function*>& coefficients, 
                                                      const MeshFunction<uint>* exterior_facet_domains); 


    static void compute_tensor_on_one_interior_facet(const Form& a,
                                                      UFC& ufc, 
                                                      const Cell& cell1, 
                                                      const Cell& cell2, 
                                                      const Facet& facet,
                                                      const std::vector<const Function*>& coefficients, 
                                                      const MeshFunction<uint>* exterior_facet_domains); 

    static void apply_bc(double* A, double* b, const uint* indicators, 
                         const double* g, uint** global_dofs, uint m, uint n); 
   
    // Class to hold data
    class Scratch
    {
      public:

      Scratch(const Form& a, const Form& L);
      ~Scratch();

      void init_cell();
      void init_macro();

      uint A_num_entries, b_num_entries;
      uint A_macro_num_entries, b_macro_num_entries;

      double* Ae;
      double* be; 
      double* Ae_macro;
      double* be_macro;
      uint* indicators;
      double* g;
    };

  };

}

#endif
