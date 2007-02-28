// Copyright (C) 2007 Anders Logg and ...
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-02-28

#ifndef __ASSEMBLER_H
#define __ASSEMBLER_H

#include <ufc.h>

#include <dolfin/DofMaps.h>

namespace dolfin
{

  class GenericTensor;
  class Mesh;
  class AssemblyData;

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
    void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh);

  private:
 
    // Assemble over cells
    void assembleCells(GenericTensor& A, AssemblyData& data, Mesh& mesh);

    // Assemble over exterior facets
    void assembleExteriorFacets(GenericTensor& A, AssemblyData& data, Mesh& mesh);

    // Assemble over interior facets
    void assembleInteriorFacets(GenericTensor& A, AssemblyData& data, Mesh& mesh);

    // Storage for precomputed dof maps
    DofMaps dof_maps;

  };

}

#endif
