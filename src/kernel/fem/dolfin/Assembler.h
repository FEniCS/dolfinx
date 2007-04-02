// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-04-02

#ifndef __ASSEMBLER_H
#define __ASSEMBLER_H

#include <ufc.h>

#include <dolfin/Array.h>
#include <dolfin/DofMaps.h>

namespace dolfin
{

  class GenericTensor;
  class Function;
  class NewForm;
  class Mesh;
  class UFC;

  /// This class provides automated assembly of linear systems, or
  /// more generally, assembly of a sparse tensor from a given
  /// variational form.

  class Assembler
  {
  public:

    /// Constructor
    Assembler();

    /// Destructor
    ~Assembler();

    /// Assemble tensor from given variational form and mesh
    void assemble(GenericTensor& A, const NewForm& form, Mesh& mesh);

    /// Assemble tensor from given variational form and mesh
    void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh);

    /// Assemble tensor from given variational form, mesh and coefficients
    void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh, Array<Function*> coefficients);

  private:
 
    // Assemble over cells
    void assembleCells(GenericTensor& A, Mesh& mesh, UFC& data) const;

    // Assemble over exterior facets
    void assembleExteriorFacets(GenericTensor& A, Mesh& mesh, UFC& data) const;

    // Assemble over interior facets
    void assembleInteriorFacets(GenericTensor& A, Mesh& mesh, UFC& data) const;

    // Check arguments
    void check(const ufc::form& form, const Mesh& mesh, const Array<Function*>& coefficients) const;

    // Storage for dof maps
    DofMaps dof_maps;

  };

}

#endif
