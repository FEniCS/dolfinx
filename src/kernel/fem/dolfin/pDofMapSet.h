// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modifies by Magnus Vikstrøm, 2008
//
// First added:  2008-01-11
// Last changed: 2008-01-11

#ifndef __P_DOF_MAP_SET_H
#define __P_DOF_MAP_SET_H

#include <map>
#include <vector>
#include <string>
#include <ufc.h>

#include <dolfin/constants.h>
#include <dolfin/pDofMap.h>

namespace dolfin
{

  class pForm;
  class Mesh;
  class pUFC;

  /// This class provides storage and caching of (precomputed) dof
  /// maps and enables reuse of already computed dof maps with equal
  /// signatures.

  class pDofMapSet
  {
  public:
    
    /// Create empty set of dof maps
    pDofMapSet();

    /// Create set of dof maps
    pDofMapSet(const pForm& form, Mesh& mesh);

    /// Create set of dof maps
    pDofMapSet(const ufc::form& form, Mesh& mesh);

    /// Destructor
    ~pDofMapSet();

    /// Update set of dof maps for given form
    void update(const pForm& form, Mesh& mesh);

    /// Update set of dof maps for given form
    void update(const ufc::form& form, Mesh& mesh);
    
    /// Return number of dof maps
    uint size() const;
    
    /// Return dof map for argument function i
    pDofMap& operator[] (uint i) const;
    
  private:

    // Cached precomputed dof maps
    std::map<const std::string, std::pair<ufc::dof_map*, pDofMap*> > dof_map_cache;

    // Array of dof maps for current form
    std::vector<pDofMap*> dof_map_set;

    // Iterator for map
    typedef std::map<const std::string, std::pair<ufc::dof_map*, pDofMap*> >::iterator map_iterator;

  };

}

#endif
