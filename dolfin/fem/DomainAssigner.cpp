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

#include <dolfin/common/NoDeleter.h>
#include "Form.h"
#include "DomainAssigner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const CellDomainAssigner&
CellDomainAssigner::operator= (std::shared_ptr<const MeshFunction<std::size_t>> domains)
{
  _form.set_cell_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
const ExteriorFacetDomainAssigner&
ExteriorFacetDomainAssigner::operator= (std::shared_ptr<const MeshFunction<std::size_t>> domains)
{
  _form.set_exterior_facet_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
const InteriorFacetDomainAssigner&
InteriorFacetDomainAssigner::operator= (std::shared_ptr<const MeshFunction<std::size_t>> domains)
{
  _form.set_interior_facet_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
const VertexDomainAssigner&
VertexDomainAssigner::operator= (std::shared_ptr<const MeshFunction<std::size_t>> domains)
{
  _form.set_vertex_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
