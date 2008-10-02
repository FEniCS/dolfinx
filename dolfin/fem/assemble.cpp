// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-01-17
// Last changed: 2008-08-21

#include "Assembler.h"
#include "assemble.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, Form& form, Mesh& mesh,
                      bool reset_tensor)
{
  Assembler assembler(mesh);
  assembler.assemble(A, form, reset_tensor);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                      DirichletBC& bc, Mesh& mesh, bool reset_tensor)
{
  Assembler assembler(mesh);
  assembler.assemble(A, a, b, L, bc, reset_tensor); 
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericMatrix& A, Form& a, GenericVector& b, Form& L, 
                         Array<DirichletBC*>& bcs, Mesh& mesh, bool reset_tensor)
{
  Assembler assembler(mesh);
  assembler.assemble(A, a, b, L, bcs, reset_tensor); 
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, Form& form, Mesh& mesh,
                      const SubDomain& sub_domain,
                      bool reset_tensor)
{
  Assembler assembler(mesh);
  assembler.assemble(A, form, sub_domain, reset_tensor);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, Form& form, Mesh& mesh,
                      const MeshFunction<dolfin::uint>& cell_domains,
                      const MeshFunction<dolfin::uint>& exterior_facet_domains,
                      const MeshFunction<dolfin::uint>& interior_facet_domains,
                      bool reset_tensor)
{
  Assembler assembler(mesh);
  assembler.assemble(A, form,
                     cell_domains,
                     exterior_facet_domains,
                     interior_facet_domains,
                     reset_tensor);
}
//-----------------------------------------------------------------------------
double dolfin::assemble(Form& form, Mesh& mesh,
                              bool reset_tensor)
{
  Assembler assembler(mesh);
  return assembler.assemble(form, reset_tensor);
}
//-----------------------------------------------------------------------------
double dolfin::assemble(Form& form, Mesh& mesh, const SubDomain& sub_domain,
                              bool reset_tensor)
{
  Assembler assembler(mesh);
  return assembler.assemble(form, sub_domain, reset_tensor);
}
//-----------------------------------------------------------------------------
double dolfin::assemble(Form& form, Mesh& mesh,
                              const MeshFunction<dolfin::uint>& cell_domains,
                              const MeshFunction<dolfin::uint>& exterior_facet_domains,
                              const MeshFunction<dolfin::uint>& interior_facet_domains,
                              bool reset_tensor)
{
  Assembler assembler(mesh);
  return assembler.assemble(form,
                            cell_domains,
                            exterior_facet_domains,
                            interior_facet_domains,
                            reset_tensor);
}
//----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A, const ufc::form& form, Mesh& mesh, 
                      Array<Function*>& coefficients,
                      DofMapSet& dof_map_set,
                      const MeshFunction<uint>* cell_domains,
                      const MeshFunction<uint>* exterior_facet_domains,
                      const MeshFunction<uint>* interior_facet_domains,
                      bool reset_tensor)
{
  Assembler assembler(mesh);
  assembler.assemble(A, form, coefficients, dof_map_set,
                     cell_domains, exterior_facet_domains, interior_facet_domains,
                     reset_tensor);
}
//----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A, const ufc::form& A_form, 
                             const Array<Function*>& A_coefficients, const DofMapSet& A_dof_map_set,
                             GenericVector& b, const ufc::form& b_form, 
                             const Array<Function*>& b_coefficients, const DofMapSet& b_dof_map_set,
                             const GenericVector* x0,
                             Mesh& mesh, 
                             Array<DirichletBC*>& bcs, const MeshFunction<uint>* cell_domains, 
                             const MeshFunction<uint>* exterior_facet_domains,
                             const MeshFunction<uint>* interior_facet_domains,
                             bool reset_tensors)
{
  Assembler assembler(mesh);
  assembler.assemble_system(A, A_form, A_coefficients, A_dof_map_set,
                            b, b_form, b_coefficients, b_dof_map_set,
                            x0, bcs, cell_domains, exterior_facet_domains, 
                            interior_facet_domains, reset_tensors);
}
//----------------------------------------------------------------------------
