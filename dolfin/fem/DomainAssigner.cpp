// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-03-09
// Last changed: 2011-03-10

#include <dolfin/common/NoDeleter.h>
#include "Form.h"
#include "DomainAssigner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DomainAssigner::DomainAssigner(Form& form) : form(form)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DomainAssigner::~DomainAssigner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const DomainAssigner&
DomainAssigner::operator= (const MeshFunction<uint>& domains)
{
  assign(reference_to_no_delete_pointer(domains));
  return *this;
}
//-----------------------------------------------------------------------------
const DomainAssigner&
DomainAssigner::operator= (boost::shared_ptr<const MeshFunction<uint> > domains)
{
  assign(domains);
  return *this;
}
//-----------------------------------------------------------------------------
void CellDomainAssigner::assign
(boost::shared_ptr<const MeshFunction<uint> > domains)
{
  form.set_cell_domains(domains);
}
//-----------------------------------------------------------------------------
void ExteriorFacetDomainAssigner::assign
(boost::shared_ptr<const MeshFunction<uint> > domains)
{
  form.set_exterior_facet_domains(domains);
}
//-----------------------------------------------------------------------------
void InteriorFacetDomainAssigner::assign
(boost::shared_ptr<const MeshFunction<uint> > domains)
{
  form.set_interior_facet_domains(domains);
}
//-----------------------------------------------------------------------------
