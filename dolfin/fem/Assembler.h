// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007-2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-11-16

#ifndef __ASSEMBLER_H
#define __ASSEMBLER_H

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

  /// This class provides automated assembly of linear systems, or
  /// more generally, assembly of a sparse tensor from a given
  /// variational form.
  ///
  /// The MeshFunction arguments can be used to specify assembly over
  /// subdomains of the mesh cells, exterior facets or interior
  /// facets. Either a null pointer or an empty MeshFunction may be
  /// used to specify that the tensor should be assembled over the
  /// entire set of cells or facets.
  ///
  /// Note that the assemble_system() functions apply boundary
  /// conditions symmetrically.

  class Assembler
  {
  public:

    /// Assemble tensor
    static void assemble(GenericTensor& A,
                         const Form& a,
                         bool reset_tensor=true);

    /// Assemble tensor on sub domain
    static void assemble(GenericTensor& A,
                         const Form& a,
                         const SubDomain& sub_domain,
                         bool reset_tensor=true);

    /// Assemble tensor on sub domains
    static void assemble(GenericTensor& A,
                         const Form& a,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_tensor=true);

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


    static void compute_mesh_function_from_mesh_arrays(Mesh& mesh);

  private:

    // Assemble over cells
    static void assemble_cells(GenericTensor& A,
                               const Form& a,
                               UFC& data,
                               const MeshFunction<uint>* domains,
                               std::vector<double>* values);

    // Assemble over exterior facets
    static void assemble_exterior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& data,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values);

    // Assemble over interior facets
    static void assemble_interior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& data,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values);

    // Check form
    static void check(const Form& a);

    // Initialize global tensor
    static void init_global_tensor(GenericTensor& A,
                                   const Form& a,
                                   UFC& ufc,
                                   bool reset_tensor);

    // Pretty-printing for progress bar
    static std::string progress_message(uint rank,
                                        std::string integral_type);

    static void compute_tensor_on_one_cell(const Form& a,
                                    UFC& ufc, 
                                    const Cell& cell, 
                                    const std::vector<const Function*>& coefficients, 
                                    const MeshFunction<uint>* cell_domains
                                    ); 
    
    static void compute_tensor_on_one_exterior_facet (const Form& a,
                                               UFC& ufc, 
                                               const Cell& cell, 
                                               const Facet& facet,
                                               const std::vector<const Function*>& coefficients, 
                                               const MeshFunction<uint>* exterior_facet_domains
                                               ); 


    static void compute_tensor_on_one_interior_facet (const Form& a,
                                               UFC& ufc, 
                                               const Cell& cell1, 
                                               const Cell& cell2, 
                                               const Facet& facet,
                                               const std::vector<const Function*>& coefficients, 
                                               const MeshFunction<uint>* exterior_facet_domains
                                               ); 




  };

}

#endif
