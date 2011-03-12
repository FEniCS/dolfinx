// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-03-09
// Last changed: 2011-03-11

#include <dolfin/common/NoDeleter.h>
#include "Form.h"
#include "DomainAssigner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const CellDomainAssigner&
CellDomainAssigner::operator= (const MeshFunction<uint>& domains)
{
  form.set_cell_domains(reference_to_no_delete_pointer(domains));
  return *this;
}
//-----------------------------------------------------------------------------
const CellDomainAssigner&
CellDomainAssigner::operator= (boost::shared_ptr<const MeshFunction<uint> > domains)
{
  form.set_cell_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
const ExteriorFacetDomainAssigner&
ExteriorFacetDomainAssigner::operator= (const MeshFunction<uint>& domains)
{
  form.set_exterior_facet_domains(reference_to_no_delete_pointer(domains));
  return *this;
}
//-----------------------------------------------------------------------------
const ExteriorFacetDomainAssigner&
ExteriorFacetDomainAssigner::operator= (boost::shared_ptr<const MeshFunction<uint> > domains)
{
  form.set_exterior_facet_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
const InteriorFacetDomainAssigner&
InteriorFacetDomainAssigner::operator= (const MeshFunction<uint>& domains)
{
  form.set_interior_facet_domains(reference_to_no_delete_pointer(domains));
  return *this;
}
//-----------------------------------------------------------------------------
const InteriorFacetDomainAssigner&
InteriorFacetDomainAssigner::operator= (boost::shared_ptr<const MeshFunction<uint> > domains)
{
  form.set_interior_facet_domains(domains);
  return *this;
}
//-----------------------------------------------------------------------------
