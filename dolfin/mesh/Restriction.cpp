// Copyright (C) 2012 Anders Logg
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
// First added:  2012-11-14
// Last changed: 2012-11-19

#include <dolfin/common/NoDeleter.h>
#include "SubDomain.h"
#include "Restriction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Restriction::Restriction(const Mesh& mesh,
                         const SubDomain& sub_domain)
{
  init_from_subdomain(mesh, sub_domain, mesh.topology().dim());
}
//-----------------------------------------------------------------------------
Restriction::Restriction(const Mesh& mesh,
                         const SubDomain& sub_domain, uint dim)
{
  init_from_subdomain(mesh, sub_domain, dim);
}
//-----------------------------------------------------------------------------
Restriction::Restriction(const MeshFunction<std::size_t>& domain_markers,
                         uint domain_number)
  : _domain_markers(reference_to_no_delete_pointer(domain_markers)),
    _domain_number(domain_number)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Restriction::Restriction(boost::shared_ptr<const MeshFunction<std::size_t> > domain_markers,
                         uint domain_number)
  : _domain_markers(domain_markers),
    _domain_number(domain_number)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Mesh& Restriction::mesh() const
{
  dolfin_assert(_domain_markers);
  return _domain_markers->mesh();
}
//-----------------------------------------------------------------------------
dolfin::uint Restriction::dim() const
{
  dolfin_assert(_domain_markers);
  return _domain_markers->dim();
}
//-----------------------------------------------------------------------------
void Restriction::init_from_subdomain(const Mesh& mesh,
                                      const SubDomain& sub_domain, uint dim)
{
  // Create mesh function
  MeshFunction<std::size_t>* __domain_markers = new MeshFunction<std::size_t>(mesh, dim);
  dolfin_assert(__domain_markers);

  // Set all markers to 1 and mark current domain as 0
  __domain_markers->set_all(1);
  sub_domain.mark(*__domain_markers, 0);
  _domain_number = 0;

  // Store shared pointer
  _domain_markers = boost::shared_ptr<const MeshFunction<std::size_t> >(__domain_markers);
}
//-----------------------------------------------------------------------------
