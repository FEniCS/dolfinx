// Copyright (C) 2011 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-03-09
// Last changed: 2011-03-11

#ifndef __DOMAIN_ASSIGNER_H
#define __DOMAIN_ASSIGNER_H

#include <boost/shared_ptr.hpp>

namespace dolfin
{

  class Form;
  template <class T> class MeshFunction;

  /// These classes are used for assignment of domains to forms:
  ///
  ///   a.cell_domains = cell_domains
  ///   a.exterior_facet_domains = exterior_facet_domains
  ///   a.interior_facet_domains = interior_facet_domains
  ///
  /// where the arguments can be either objects/references or
  /// shared pointers to MeshFunctions.

  /// Assignment of cell domains
  class CellDomainAssigner
  {
  public:

    // Constructor
    CellDomainAssigner(Form& form) : form(form) {}

    // Assign reference
    const CellDomainAssigner& operator= (const MeshFunction<uint>& domains);

    // Assign shared pointer
    const CellDomainAssigner& operator= (boost::shared_ptr<const MeshFunction<uint> > domains);

  private:

    // The form
    Form& form;

  };

  /// Assignment of exterior facet domains
  class ExteriorFacetDomainAssigner
  {
  public:

    // Constructor
    ExteriorFacetDomainAssigner(Form& form) : form(form) {}

    // Assign reference
    const ExteriorFacetDomainAssigner& operator= (const MeshFunction<uint>& domains);

    // Assign shared pointer
    const ExteriorFacetDomainAssigner& operator= (boost::shared_ptr<const MeshFunction<uint> > domains);

  private:

    // The form
    Form& form;

  };

  /// Assignment of interior facet domains
  class InteriorFacetDomainAssigner
  {
  public:

    // Constructor
    InteriorFacetDomainAssigner(Form& form) : form(form) {}

    // Assign reference
    const InteriorFacetDomainAssigner& operator= (const MeshFunction<uint>& domains);

    // Assign shared pointer
    const InteriorFacetDomainAssigner& operator= (boost::shared_ptr<const MeshFunction<uint> > domains);

  private:

    // The form
    Form& form;

  };

}

#endif
