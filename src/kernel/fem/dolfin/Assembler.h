// Copyright (C) 2007 Anders Logg and ...
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-03-01

#ifndef __ASSEMBLER_H
#define __ASSEMBLER_H

#include <ufc.h>

#include <dolfin/DofMaps.h>

namespace dolfin
{

  class GenericTensor;
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
    void assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh);

  private:
 
    // Assemble over cells
    void assembleCells(GenericTensor& A, Mesh& mesh, UFC& data) const;

    // Assemble over exterior facets
    void assembleExteriorFacets(GenericTensor& A, Mesh& mesh, UFC& data) const;

    // Assemble over interior facets
    void assembleInteriorFacets(GenericTensor& A, Mesh& mesh, UFC& data) const;

    // Initialize mesh entities used by dof maps
    void initMeshEntities(Mesh& mesh, UFC& data) const;

    // Initialize global tensor
    void initGlobalTensor(GenericTensor& A, Mesh& mesh, UFC& data, DofMaps& dof_maps) const;

    // Storage for dof maps
    DofMaps dof_maps;

  };

}

#endif
