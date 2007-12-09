// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-17
// Last changed: 2007-08-28

#include <dolfin/Assembler.h>
#include <dolfin/assemble.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const Form& form, Mesh& mesh, DofMapSet& dof_map_set)
{
  Assembler assembler(mesh, dof_map_set);
  assembler.assemble(A, form);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const Form& form, Mesh& mesh, DofMapSet& dof_map_set,
                      const SubDomain& sub_domain)
{
  Assembler assembler(mesh, dof_map_set);
  assembler.assemble(A, form, sub_domain);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const Form& form, Mesh& mesh, DofMapSet& dof_map_set,
                      const MeshFunction<dolfin::uint>& cell_domains,
                      const MeshFunction<dolfin::uint>& exterior_facet_domains,
                      const MeshFunction<dolfin::uint>& interior_facet_domains)
{
  Assembler assembler(mesh, dof_map_set);
  assembler.assemble(A, form,
                     cell_domains,
                     exterior_facet_domains,
                     interior_facet_domains);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(const Form& form, Mesh& mesh, DofMapSet& dof_map_set)
{
  Assembler assembler(mesh, dof_map_set);
  return assembler.assemble(form);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(const Form& form, Mesh& mesh, DofMapSet& dof_map_set,
                              const SubDomain& sub_domain)
{
  Assembler assembler(mesh, dof_map_set);
  return assembler.assemble(form, sub_domain);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(const Form& form, Mesh& mesh, DofMapSet& dof_map_set,
                              const MeshFunction<dolfin::uint>& cell_domains,
                              const MeshFunction<dolfin::uint>& exterior_facet_domains,
                              const MeshFunction<dolfin::uint>& interior_facet_domains)
{
  Assembler assembler(mesh, dof_map_set);
  return assembler.assemble(form,
                            cell_domains,
                            exterior_facet_domains,
                            interior_facet_domains);
}
//----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh, DofMapSet& dof_map_set,
                      Array<Function*>& coefficients,
                      const MeshFunction<uint>* cell_domains,
                      const MeshFunction<uint>* exterior_facet_domains,
                      const MeshFunction<uint>* interior_facet_domains, bool reset_tensor)
{
  Assembler assembler(mesh, dof_map_set);
  assembler.assemble(A, form, coefficients,
                     cell_domains, exterior_facet_domains, interior_facet_domains,
                     reset_tensor);
}
//----------------------------------------------------------------------------
