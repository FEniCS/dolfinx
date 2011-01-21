// Copyright (C) 2007-2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
// Modified by Johan Hake, 2009.
//
// First added:  2007-01-17
// Last changed: 2011-01-20

#include <dolfin/la/Scalar.h>
#include "Form.h"
#include "Assembler.h"
#include "SystemAssembler.h"
#include "assemble.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A,
                      const Form& a,
                      bool reset_sparsity,
                      bool add_values)
{
  Assembler::assemble(A, a, reset_sparsity, add_values);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A,
                      const Form& a,
                      const SubDomain& sub_domain,
                      bool reset_sparsity,
                      bool add_values)
{
  Assembler::assemble(A, a, sub_domain, reset_sparsity, add_values);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A,
                      const Form& a,
                      const MeshFunction<uint>* cell_domains,
                      const MeshFunction<uint>* exterior_facet_domains,
                      const MeshFunction<uint>* interior_facet_domains,
                      bool reset_sparsity,
                      bool add_values)
{
  Assembler::assemble(A, a, cell_domains, exterior_facet_domains,
                      interior_facet_domains, reset_sparsity, add_values);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             bool reset_sparsities,
                             bool add_values)
{
  SystemAssembler::assemble(A, b, a, L, reset_sparsities, add_values);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const DirichletBC& bc,
                             bool reset_sparsities,
                             bool add_values)
{
  SystemAssembler::assemble(A, b, a, L, bc, reset_sparsities, add_values);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const std::vector<const DirichletBC*>& bcs,
                             bool reset_sparsities,
                             bool add_values)
{
  SystemAssembler::assemble(A, b, a, L, bcs, reset_sparsities, add_values);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const std::vector<const DirichletBC*>& bcs,
                             const MeshFunction<uint>* cell_domains,
                             const MeshFunction<uint>* exterior_facet_domains,
                             const MeshFunction<uint>* interior_facet_domains,
                             const GenericVector* x0,
                             bool reset_sparsities,
                             bool add_values)
{
  SystemAssembler::assemble(A, b, a, L, bcs,
                            cell_domains, exterior_facet_domains,
                            interior_facet_domains, x0, reset_sparsities, add_values);
}
//-----------------------------------------------------------------------------
double dolfin::assemble(const Form& a,
                        bool reset_sparsity,
                        bool add_values)
{
  if (a.rank() != 0)
    error("Unable to assemble, form is not scalar.");
  Scalar s;
  Assembler::assemble(s, a, reset_sparsity, add_values);
  return s;
}
//-----------------------------------------------------------------------------
double dolfin::assemble(const Form& a,
                        const SubDomain& sub_domain,
                        bool reset_sparsity,
                        bool add_values)
{
  if (a.rank() != 0)
    error("Unable to assemble, form is not scalar.");
  Scalar s;
  Assembler::assemble(s, a, sub_domain, reset_sparsity, add_values);
  return s;
}
//-----------------------------------------------------------------------------
double dolfin::assemble(const Form& a,
                        const MeshFunction<uint>* cell_domains,
                        const MeshFunction<uint>* exterior_facet_domains,
                        const MeshFunction<uint>* interior_facet_domains,
                        bool reset_sparsity,
                        bool add_values)
{
  if (a.rank() != 0)
    error("Unable to assemble, form is not scalar.");
  Scalar s;
  Assembler::assemble(s, a, cell_domains, exterior_facet_domains,
                      interior_facet_domains, reset_sparsity, add_values);
  return s;
}
//-----------------------------------------------------------------------------
