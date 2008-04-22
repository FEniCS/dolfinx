// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modifies by Garth N. Wells, 2007.
//
// First added:  2007-01-17
// Last changed: 2007-05-24

#ifndef __DOF_MAP_SET_H
#define __DOF_MAP_SET_H

#include <map>
#include <vector>
#include <string>
#include <ufc.h>

#include <dolfin/common/types.h>
#include "DofMap.h"

namespace dolfin
{

  class Form;
  class Mesh;
  class UFC;

  /// This class provides storage and caching of (precomputed) dof
  /// maps and enables reuse of already computed dof maps with equal
  /// signatures.

  class DofMapSet
  {
  public:
    
    /// Create empty set of dof maps
    DofMapSet();

    /// Create set of dof maps
    DofMapSet(const Form& form, Mesh& mesh);

    /// Create set of dof maps (parallel)
    DofMapSet(const Form& form, Mesh& mesh, MeshFunction<uint>& partitions);

    /// Create set of dof maps
    DofMapSet(const ufc::form& form, Mesh& mesh);

    /// Create set of dof maps (parallel)
    DofMapSet(const ufc::form& form, Mesh& mesh, MeshFunction<uint>& partitions);

    /// Destructor
    ~DofMapSet();

    /// Update set of dof maps for given form
    void update(const Form& form, Mesh& mesh);

    /// Update set of dof maps for given form (parallel)
    void update(const Form& form, Mesh& mesh, MeshFunction<uint>& partitions);

    /// Update set of dof maps for given form
    void update(const ufc::form& form, Mesh& mesh);
    
    /// Update set of dof maps for given form (parallel)
    void update(const ufc::form& form, Mesh& mesh, MeshFunction<uint>& partitions);

    /// Return number of dof maps
    uint size() const;
    
    /// Return dof map for argument function i
    DofMap& operator[] (uint i) const;
    
    /// Build parallel dof maps
    void build(UFC& ufc) const;

    bool parallel() const
      { return _parallel; }

  private:

    // Cached precomputed dof maps
    std::map<const std::string, std::pair<ufc::dof_map*, DofMap*> > dof_map_cache;

    // Array of dof maps for current form
    std::vector<DofMap*> dof_map_set;

    // Iterator for map
    typedef std::map<const std::string, std::pair<ufc::dof_map*, DofMap*> >::iterator map_iterator;

    bool _parallel;

  };

}

#endif
