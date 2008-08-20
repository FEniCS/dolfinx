// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007, 2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-08-20

#ifndef __ASSEMBLER_H
#define __ASSEMBLER_H

#include <dolfin/common/types.h>

// Forward declaration
namespace ufc 
{
  class form; 
}

namespace dolfin
{

  // Forward declarations
  class DirichletBC;
  class DofMapSet;
  class GenericMatrix;
  class GenericTensor;
  class GenericVector;
  class Function;
  class Form;
  class Mesh;
  class SubDomain;
  class UFC;
  template<class T> class Array;
  template<class T> class MeshFunction;

  /// This class provides automated assembly of linear systems, or
  /// more generally, assembly of a sparse tensor from a given
  /// variational form.

  class Assembler
  {
  public:

    /// Constructor
    Assembler(Mesh& mesh);

    /// Destructor
    ~Assembler();

    /// Assemble tensor from given variational form
    void assemble(GenericTensor& A, Form& form,
                  bool reset_tensor=true);

    /// Assemble system (A, b) and apply Dirichlet boundary conditions from 
    /// given variational forms
    void assemble(GenericMatrix& A, Form& A_form, GenericVector& b, 
                  Form& b_form, Array<DirichletBC*>& bcs, bool reset_tensor=true);

    /// Assemble tensor from given variational form over a sub domain
    void assemble(GenericTensor& A, Form& form, const SubDomain& sub_domain,
                  bool reset_tensor=true);

    /// Assemble tensor from given variational form over a sub domain
    //void assemble(GenericTensor& A, Form& form,
    //              const MeshFunction<uint>& domains, uint domain, bool reset_tensor = true);

    /// Assemble tensor from given variational form over sub domains
    void assemble(GenericTensor& A, Form& form,
                  const MeshFunction<uint>& cell_domains,
                  const MeshFunction<uint>& exterior_facet_domains,
                  const MeshFunction<uint>& interior_facet_domains,
                  bool reset_tensor=true);
    
    /// Assemble scalar from given variational form
    real assemble(Form& form, bool reset_tensor=true);
    
    /// Assemble scalar from given variational form over a sub domain
    real assemble(Form& form, const SubDomain& sub_domain, bool reset_tensor);
    
    /// Assemble scalar from given variational form over sub domains
    real assemble(Form& form,
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
    void assemble(GenericTensor& A, const ufc::form& form,
                  const Array<Function*>& coefficients,
                  const DofMapSet& dof_map_set,
                  const MeshFunction<uint>* cell_domains,
                  const MeshFunction<uint>* exterior_facet_domains,
                  const MeshFunction<uint>* interior_facet_domains,
                  bool reset_tensor = true);

    /// Assemble linear system Ax = b and enforce Dirichlet conditions.  
    //  Notice that the Dirichlet conditions are enforced in a symmetric way.  
    void assemble_system(GenericMatrix& A, const ufc::form& A_form, 
                         const Array<Function*>& A_coefficients, const DofMapSet& A_dof_map_set,
                         GenericVector& b, const ufc::form& b_form, 
                         const Array<Function*>& b_coefficients, const DofMapSet& b_dof_map_set,
                         const GenericVector* x0,
                         DirichletBC& bc, const MeshFunction<uint>* cell_domains, 
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_tensors=true);

    void assemble_system(GenericMatrix& A, const ufc::form& A_form, 
                         const Array<Function*>& A_coefficients, const DofMapSet& A_dof_map_set,
                         GenericVector& b, const ufc::form& b_form, 
                         const Array<Function*>& b_coefficients, const DofMapSet& b_dof_map_set,
                         const GenericVector* x0,
                         Array<DirichletBC*> bcs, const MeshFunction<uint>* cell_domains, 
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_tensors=true);


  private:
 
    // Assemble over cells
    void assembleCells(GenericTensor& A,
                       const Array<Function*>& coefficients,
                       const DofMapSet& dof_set_map,
                       UFC& data,
                       const MeshFunction<uint>* domains) const;

    // Assemble over exterior facets
    void assembleExteriorFacets(GenericTensor& A,
                                const Array<Function*>& coefficients,
                                const DofMapSet& dof_set_map,
                                UFC& data,
                                const MeshFunction<uint>* domains) const;

    // Assemble over interior facets
    void assembleInteriorFacets(GenericTensor& A,
                                const Array<Function*>& coefficients,
                                const DofMapSet& dof_set_map,
                                UFC& data,
                                const MeshFunction<uint>* domains) const;

    // Check arguments
    void check(const ufc::form& form,
               const Array<Function*>& coefficients,
               const Mesh& mesh) const;

    // Initialize global tensor
    void initGlobalTensor(GenericTensor& A, const DofMapSet& dof_map_set, UFC& ufc, bool reset_tensor) const;

    // Pretty-printing for progress bar
    std::string progressMessage(uint rank, std::string integral_type) const;

    // The mesh
    Mesh& mesh;

    // Are we running in parallel?
    bool parallel;

  };

}

#endif
