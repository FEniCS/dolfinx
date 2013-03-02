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
// Last changed: 2011-03-11

#include <dolfin/common/NoDeleter.h>
#include "Form.h"
#include "DomainAssigner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const CellDomainAssigner&
CellDomainAssigner::operator= (const MeshFunction<std::size_t>& domains)
{
  _form.set_cell_domains(reference_to_no_delete_pointer(domains));
  return *this;
}
//-----------------------------------------------------------------------------
const CellDomainAssigner&
CellDomainAssigner::operator= (boost::shared_ptr<const MeshFunction<std::size_t> > domains)
{
  _form.set_cell_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
const ExteriorFacetDomainAssigner&
ExteriorFacetDomainAssigner::operator= (const MeshFunction<std::size_t>& domains)
{
  _form.set_exterior_facet_domains(reference_to_no_delete_pointer(domains));
  return *this;
}
//-----------------------------------------------------------------------------
const ExteriorFacetDomainAssigner&
ExteriorFacetDomainAssigner::operator= (boost::shared_ptr<const MeshFunction<std::size_t> > domains)
{
  _form.set_exterior_facet_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
const InteriorFacetDomainAssigner&
InteriorFacetDomainAssigner::operator= (const MeshFunction<std::size_t>& domains)
{
  _form.set_interior_facet_domains(reference_to_no_delete_pointer(domains));
  return *this;
}
//-----------------------------------------------------------------------------
const InteriorFacetDomainAssigner&
InteriorFacetDomainAssigner::operator= (boost::shared_ptr<const MeshFunction<std::size_t> > domains)
{
  _form.set_interior_facet_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
