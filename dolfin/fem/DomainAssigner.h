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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-03-09
// Last changed: 2014-10-03

#ifndef __DOMAIN_ASSIGNER_H
#define __DOMAIN_ASSIGNER_H

#include <memory>

namespace dolfin
{

  class Form;
  template <typename T> class MeshFunction;

  /// These classes are used for assignment of domains to forms:
  ///
  ///   a.dx = cell_domains
  ///   a.ds = exterior_facet_domains
  ///   a.dS = interior_facet_domains
  ///
  /// where the arguments can be either objects/references or
  /// shared pointers to MeshFunctions.

  /// Assignment of cell domains
  class CellDomainAssigner
  {
  public:

    // Constructor
    explicit CellDomainAssigner(Form& form) : _form(form) {}

    // Assign shared pointer
    const CellDomainAssigner& operator=
    (std::shared_ptr<const MeshFunction<std::size_t>> domains);

  private:

    // The form
    Form& _form;

  };

  /// Assignment of exterior facet domains
  class ExteriorFacetDomainAssigner
  {
  public:

    // Constructor
    explicit ExteriorFacetDomainAssigner(Form& form) : _form(form) {}

    // Assign shared pointer
    const ExteriorFacetDomainAssigner& operator=
    (std::shared_ptr<const MeshFunction<std::size_t>> domains);

  private:

    // The form
    Form& _form;

  };

  /// Assignment of interior facet domains
  class InteriorFacetDomainAssigner
  {
  public:

    // Constructor
    explicit InteriorFacetDomainAssigner(Form& form) : _form(form) {}

    // Assign shared pointer
    const InteriorFacetDomainAssigner& operator=
    (std::shared_ptr<const MeshFunction<std::size_t>> domains);

  private:

    // The form
    Form& _form;

  };

  /// Assignment of vertex domains
  class VertexDomainAssigner
  {
  public:

    // Constructor
    explicit VertexDomainAssigner(Form& form) : _form(form) {}

    // Assign shared pointer
    const VertexDomainAssigner& operator=
    (std::shared_ptr<const MeshFunction<std::size_t>> domains);

  private:

    // The form
    Form& _form;

  };

}

#endif
