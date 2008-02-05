// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007
// Modified by Magnus Vikstrøm, 2008
//
// First added:  2007-01-17
// Last changed: 2008-01-15

#ifndef __P_ASSEMBLER_H
#define __P_ASSEMBLER_H

#include <ufc.h>

#include <dolfin/Array.h>
#include <dolfin/MeshFunction.h>

namespace dolfin
{

  class DofMapSet;
  class GenericTensor;
  class Function;
  class Form;
  class Mesh;
  class SubDomain;
  class UFC;

  /// This class provides automated assembly of linear systems, or
  /// more generally, assembly of a sparse tensor from a given
  /// variational form.

  class pAssembler
  {
  public:

    /// Constructor
    pAssembler(Mesh& mesh);

    /// Constructor
    pAssembler(Mesh& mesh, MeshFunction<uint>& partitions);

    /// Destructor
    ~pAssembler();

    /// Assemble tensor from given variational form
    void assemble(GenericTensor& A, Form& form, bool reset_tensor = true);

    /// Assemble tensor from given variational form over a sub domain
    void assemble(GenericTensor& A, Form& form,
                  const SubDomain& sub_domain, bool reset_tensor = true);

    /// Assemble tensor from given variational form over sub domains
    void assemble(GenericTensor& A, Form& form,
                  const MeshFunction<uint>& cell_domains,
                  const MeshFunction<uint>& exterior_facet_domains,
                  const MeshFunction<uint>& interior_facet_domains, bool reset_tensor = true);
    
    /// Assemble scalar from given variational form
    real assemble(Form& form);
    
    /// Assemble scalar from given variational form over a sub domain
    real assemble(Form& form, const SubDomain& sub_domain);
    
    /// Assemble scalar from given variational form over sub domains
    real assemble(Form& form,
                  const MeshFunction<uint>& cell_domains,
                  const MeshFunction<uint>& exterior_facet_domains,
                  const MeshFunction<uint>& interior_facet_domains);
    
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
                  const MeshFunction<uint>* interior_facet_domains, bool reset_tensor = true);
      
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
               const Array<Function*>& coefficients) const;

    // Initialize global tensor
    void initGlobalTensor(GenericTensor& A, const DofMapSet& dof_map_set, UFC& ufc, bool reset_tensor) const;

    // The mesh
    Mesh& mesh;

    // The partitions;
    MeshFunction<uint>* partitions;

  };

}

#endif
