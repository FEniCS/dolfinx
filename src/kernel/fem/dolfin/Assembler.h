// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-17
// Last changed: 2007-04-30

#ifndef __ASSEMBLER_H
#define __ASSEMBLER_H

#include <ufc.h>

#include <dolfin/Array.h>
#include <dolfin/DofMaps.h>

namespace dolfin
{

  class GenericTensor;
  class Function;
  class Form;
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
    void assemble(GenericTensor& A, const Form& form, Mesh& mesh);

    /// Assemble tensor from given variational form and mesh
    void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh);

    /// Assemble tensor from given variational form, mesh and coefficients
    void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh,
                  Array<Function*> coefficients);

    /// Assemble scalar from given variational form and mesh
    real assemble(const Form& form, Mesh& mesh);

    /// Assemble scalar from given variational form and mesh
    real assemble(const ufc::form& form, Mesh& mesh);

    /// Assemble scalar from given variational form, mesh and coefficients
    real assemble(const ufc::form& form, Mesh& mesh,
                  Array<Function*> coefficients);

  private:
 
    // Assemble over cells
    void assembleCells(GenericTensor& A, Mesh& mesh,
                       Array<Function*>& coefficients,
                       UFC& data) const;

    // Assemble over exterior facets
    void assembleExteriorFacets(GenericTensor& A, Mesh& mesh,
                                Array<Function*>& coefficients,
                                UFC& data) const;

    // Assemble over interior facets
    void assembleInteriorFacets(GenericTensor& A, Mesh& mesh,
                                Array<Function*>& coefficients,
                                UFC& data) const;

    // Check arguments
    void check(const ufc::form& form, const Mesh& mesh,
               Array<Function*>& coefficients) const;

    // Initialize global tensor
    void initGlobalTensor(GenericTensor& A, const UFC& ufc) const;

    // Initialize coefficients
    void initCoefficients(Array<Function*>& coefficients, const UFC& ufc) const;

    // Storage for dof maps
    DofMaps dof_maps;

  };

}

#endif
