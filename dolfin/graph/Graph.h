// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-16
// Last changed: 2010-11-17

#ifndef __GRAPH_TYPES_H
#define __GRAPH_TYPES_H

#include <vector>
#include <boost/unordered_set.hpp>
#include <dolfin/common/Set.h>

namespace dolfin
{

  /// Typedefs for simple graph data structures

  /// Vector of unordered sets
  //typedef std::vector<boost::unordered_set<unsigned int> > Graph;
  typedef std::vector<dolfin::Set<unsigned int> > Graph;

}

#endif
