// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARTITION_H
#define __PARTITION_H

#include <dolfin/constants.h>
#include <dolfin/ShortList.h>

namespace dolfin {

  class TimeSlabData;

  /// Partition is used in the recursive construction of time slabs
  /// and contains a list of component indices. The order of these
  /// indices determine which components should be in which time slab.

  class Partition {
  public:

    /// Constructor
    Partition(int N, real timestep);

    /// Destructor
    ~Partition();

    /// Return size of partition
    int size() const;

    /// Return component index number i
    int index(int i) const;

    /// Update time steps
    void update(TimeSlabData& data, int offset);
    
    /// Partition (reorder) components
    void partition(int offset, int& end, real& K);

  private:

    class Component {
    public:

      Component();
      ~Component();

      void operator=(int zero);

      int index;
      real timestep;

    };
    
    real maximum(int offset);

    // List of components
    ShortList<Component> components;

  };

}

#endif
