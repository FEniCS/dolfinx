// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-17
// Last changed: 2007-08-28

#include "Assembler.h"
#include "assemble.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, Form& form, Mesh& mesh)
{
  Assembler assembler(mesh);
  assembler.assemble(A, form);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, Form& form, Mesh& mesh,
                      const SubDomain& sub_domain)
{
  Assembler assembler(mesh);
  assembler.assemble(A, form, sub_domain);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, Form& form, Mesh& mesh,
                      const MeshFunction<dolfin::uint>& cell_domains,
                      const MeshFunction<dolfin::uint>& exterior_facet_domains,
                      const MeshFunction<dolfin::uint>& interior_facet_domains)
{
  Assembler assembler(mesh);
  assembler.assemble(A, form,
                     cell_domains,
                     exterior_facet_domains,
                     interior_facet_domains);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(Form& form, Mesh& mesh)
{
  Assembler assembler(mesh);
  return assembler.assemble(form);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(Form& form, Mesh& mesh,
                              const SubDomain& sub_domain)
{
  Assembler assembler(mesh);
  return assembler.assemble(form, sub_domain);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(Form& form, Mesh& mesh,
                              const MeshFunction<dolfin::uint>& cell_domains,
                              const MeshFunction<dolfin::uint>& exterior_facet_domains,
                              const MeshFunction<dolfin::uint>& interior_facet_domains)
{
  Assembler assembler(mesh);
  return assembler.assemble(form,
                            cell_domains,
                            exterior_facet_domains,
                            interior_facet_domains);
}
//----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh, 
                      Array<Function*>& coefficients,
                      DofMapSet& dof_map_set,
                      const MeshFunction<uint>* cell_domains,
                      const MeshFunction<uint>* exterior_facet_domains,
                      const MeshFunction<uint>* interior_facet_domains, bool reset_tensor)
{
  Assembler assembler(mesh);
  assembler.assemble(A, form, coefficients, dof_map_set,
                     cell_domains, exterior_facet_domains, interior_facet_domains,
                     reset_tensor);
}
//----------------------------------------------------------------------------
