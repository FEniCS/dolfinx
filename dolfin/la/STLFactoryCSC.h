// Copyright (C) 2012 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-02-21
// Last changed:

#ifndef __DOLFIN_STL_FACTORY_CSC_H
#define __DOLFIN_STL_FACTORY_CSC_H

#include <memory>
#include "TensorLayout.h"
#include "STLFactory.h"

namespace dolfin
{

  class STLFactoryCSC : public STLFactory
  {
  public:

    /// Destructor
    virtual ~STLFactoryCSC() {}

    /// Create empty tensor layout
    virtual std::shared_ptr<TensorLayout> create_layout(std::size_t rank) const
    {
      std::shared_ptr<TensorLayout>
        pattern(new TensorLayout(1, TensorLayout::Sparsity::DENSE));
      return pattern;
    }

    /// Return singleton instance
    static STLFactoryCSC& instance()
    { return factory; }

  private:

    /// Private Constructor
    STLFactoryCSC() {}

    // Singleton instance
    static STLFactoryCSC factory;

  };
}

#endif
