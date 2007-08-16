// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-17
// Last changed: 2007-05-14

#include <dolfin/Assembler.h>
#include <dolfin/assemble.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const Form& form, Mesh& mesh)
{
  Assembler assembler;
  assembler.assemble(A, form, mesh);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const Form& form, Mesh& mesh,
                      const SubDomain& sub_domain)
{
  Assembler assembler;
  assembler.assemble(A, form, mesh, sub_domain);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const Form& form, Mesh& mesh, 
                      const MeshFunction<dolfin::uint>& cell_domains,
                      const MeshFunction<dolfin::uint>& exterior_facet_domains,
                      const MeshFunction<dolfin::uint>& interior_facet_domains)
{
  Assembler assembler;
  assembler.assemble(A, form, mesh,
                     cell_domains,
                     exterior_facet_domains,
                     interior_facet_domains);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(const Form& form, Mesh& mesh)
{
  Assembler assembler;
  return assembler.assemble(form, mesh);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(const Form& form, Mesh& mesh,
                              const SubDomain& sub_domain)
{
  Assembler assembler;
  return assembler.assemble(form, mesh, sub_domain);
}
//-----------------------------------------------------------------------------
dolfin::real dolfin::assemble(const Form& form, Mesh& mesh,
                              const MeshFunction<dolfin::uint>& cell_domains,
                              const MeshFunction<dolfin::uint>& exterior_facet_domains,
                              const MeshFunction<dolfin::uint>& interior_facet_domains)
{
  Assembler assembler;
  return assembler.assemble(form, mesh,
                            cell_domains,
                            exterior_facet_domains,
                            interior_facet_domains);
}
//----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh,
                      Array<Function*>& coefficients,
                      const MeshFunction<uint>* cell_domains,
                      const MeshFunction<uint>* exterior_facet_domains,
                      const MeshFunction<uint>* interior_facet_domains, bool reset_tensor)
{
  Assembler assembler;
  assembler.assemble(A, form, mesh, coefficients,
                     cell_domains, exterior_facet_domains, interior_facet_domains,
                     reset_tensor);
}
//----------------------------------------------------------------------------

// FIXME: For testing JIT compiler

void dolfin::assemble_test(GenericTensor& A, const ufc::form& form, Mesh& mesh,
                           Array<Function*>& coefficients,
                           const MeshFunction<uint>* cell_domains,
                           const MeshFunction<uint>* exterior_facet_domains,
                           const MeshFunction<uint>* interior_facet_domains, bool reset_tensor)
{
  message("assemble_test working: A, form, mesh, cell_domains, exterior_face_domains, interior_facet_domains, reset_tensor");
}
