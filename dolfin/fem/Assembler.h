// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007-2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-10-23

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
  class Function;
  class Form;
  class Mesh;
  class SubDomain;
  class UFC;
  template<class T> class MeshFunction;

  /// This class provides automated assembly of linear systems, or
  /// more generally, assembly of a sparse tensor from a given
  /// variational form.

  class Assembler
  {
  public:

    /// Assemble tensor from given variational form
    static void assemble(GenericTensor& A, Form& form, bool reset_tensor=true);

    /// Assemble system (A, b) and apply Dirichlet boundary condition from 
    /// given variational forms
    static void assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                         DirichletBC& bc, bool reset_tensor=true);

    /// Assemble system (A, b) and apply Dirichlet boundary conditions from 
    /// given variational forms
    static void assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                         std::vector<DirichletBC*>& bcs, bool reset_tensor=true);

    /// Assemble tensor from given variational form over a sub domain
    static void assemble(GenericTensor& A, Form& form, const SubDomain& sub_domain,
                         bool reset_tensor=true);

    /// Assemble tensor from given variational form over a sub domain
    //void assemble(GenericTensor& A, Form& form,
    //              const MeshFunction<uint>& domains, uint domain, bool reset_tensor = true);

    /// Assemble tensor from given variational form over sub domains
    static void assemble(GenericTensor& A, Form& form,
                         const MeshFunction<uint>& cell_domains,
                         const MeshFunction<uint>& exterior_facet_domains,
                         const MeshFunction<uint>& interior_facet_domains,
                         bool reset_tensor=true);
    
    /// Assemble scalar from given variational form
    static double assemble(Form& form, bool reset_tensor=true);
    
    /// Assemble scalar from given variational form over a sub domain
    static double assemble(Form& form, const SubDomain& sub_domain, bool reset_tensor);
    
    /// Assemble scalar from given variational form over sub domains
    static double assemble(Form& form,
                           const MeshFunction<uint>& cell_domains,
                           const MeshFunction<uint>& exterior_facet_domains,
                           const MeshFunction<uint>& interior_facet_domains,
                           bool reset_tensor);
    
    /// Assemble tensor from given (UFC) form, coefficients and sub domains.
    /// This is the main assembly function in DOLFIN. All other assembly functions
    /// end up calling this function.
    ///
    /// The MeshFunction arguments can be used to specify assembly over subdomains
    /// of the mesh cells, exterior facets and interior facets. Either a null pointer
    /// or an empty MeshFunction may be used to specify that the tensor should be
    /// assembled over the entire set of cells or facets.
    static void assemble(GenericTensor& A, const Form& form,
                         const std::vector<Function*>& coefficients,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_tensor = true);

    /// Assemble linear system Ax = b and enforce Dirichlet conditions.  
    //  Notice that the Dirichlet conditions are enforced in a symmetric way.  
    static void assemble_system(GenericMatrix& A, const Form& A_form, 
                                const std::vector<Function*>& A_coefficients,
                                GenericVector& b, const Form& b_form, 
                                const std::vector<Function*>& b_coefficients,
                                const GenericVector* x0,
                                std::vector<DirichletBC*> bcs, const MeshFunction<uint>* cell_domains, 
                                const MeshFunction<uint>* exterior_facet_domains,
                                const MeshFunction<uint>* interior_facet_domains,
                                bool reset_tensors=true);

  private:
 
    // Assemble over cells
    static void assembleCells(GenericTensor& A,
                              const Form& form,
                              const std::vector<Function*>& coefficients,
                              UFC& data,
                              const MeshFunction<uint>* domains,
                              std::vector<double>* values);
    
    // Assemble over exterior facets
    static void assembleExteriorFacets(GenericTensor& A,
                                       const Form& form,
                                       const std::vector<Function*>& coefficients,
                                       UFC& data,
                                       const MeshFunction<uint>* domains,
                                       std::vector<double>* values);

    // Assemble over interior facets
    static void assembleInteriorFacets(GenericTensor& A,
                                       const Form& form,
                                       const std::vector<Function*>& coefficients,
                                       UFC& data,
                                       const MeshFunction<uint>* domains,
                                       std::vector<double>* values);

    // Check arguments
    static void check(const Form& form,
                      const std::vector<Function*>& coefficients,
                      const Mesh& mesh);
    
    // Initialize global tensor
    static void initGlobalTensor(GenericTensor& A, const Form& form, UFC& ufc, bool reset_tensor);

    // Pretty-printing for progress bar
    static std::string progressMessage(uint rank, std::string integral_type);

  };

}

#endif
