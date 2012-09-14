// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Garth N. Wells, 2008.
// Modified by Johan Hake, 2009.
// Modified by Joachim B. Haga, 2012.
//
// First added:  2007-01-17
// Last changed: 2012-02-01

#include <dolfin/la/Scalar.h>
#include "Form.h"
#include "Assembler.h"
#include "SystemAssembler.h"
#include "SymmetricAssembler.h"
#include "assemble.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A,
                      const Form& a,
                      bool reset_sparsity,
                      bool add_values,
                      bool finalize_tensor,
                      bool keep_diagonal)
{
  Assembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(A, a);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A,
                      const Form& a,
                      const SubDomain& sub_domain,
                      bool reset_sparsity,
                      bool add_values,
                      bool finalize_tensor,
                      bool keep_diagonal)
{
  Assembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(A, a, sub_domain);
}
//-----------------------------------------------------------------------------
void dolfin::assemble(GenericTensor& A,
                      const Form& a,
                      const MeshFunction<uint>* cell_domains,
                      const MeshFunction<uint>* exterior_facet_domains,
                      const MeshFunction<uint>* interior_facet_domains,
                      bool reset_sparsity,
                      bool add_values,
                      bool finalize_tensor,
                      bool keep_diagonal)
{
  Assembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(A, a, cell_domains, exterior_facet_domains,
                      interior_facet_domains);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             bool reset_sparsity,
                             bool add_values,
                             bool finalize_tensor,
                             bool keep_diagonal)
{
  SystemAssembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(A, b, a, L);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const DirichletBC& bc,
                             bool reset_sparsity,
                             bool add_values,
                             bool finalize_tensor,
                             bool keep_diagonal)
{
  SystemAssembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(A, b, a, L, bc);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const std::vector<const DirichletBC*> bcs,
                             bool reset_sparsity,
                             bool add_values,
                             bool finalize_tensor,
                             bool keep_diagonal)
{
  SystemAssembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(A, b, a, L, bcs);
}
//-----------------------------------------------------------------------------
void dolfin::assemble_system(GenericMatrix& A,
                             GenericVector& b,
                             const Form& a,
                             const Form& L,
                             const std::vector<const DirichletBC*> bcs,
                             const MeshFunction<uint>* cell_domains,
                             const MeshFunction<uint>* exterior_facet_domains,
                             const MeshFunction<uint>* interior_facet_domains,
                             const GenericVector* x0,
                             bool reset_sparsity,
                             bool add_values,
                             bool finalize_tensor,
                             bool keep_diagonal)
{
  SystemAssembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(A, b, a, L, bcs, cell_domains, exterior_facet_domains,
                     interior_facet_domains, x0);
}
//-----------------------------------------------------------------------------
void dolfin::symmetric_assemble(GenericMatrix& As,
                                GenericMatrix& An,
                                const Form& a,
                                const std::vector<const DirichletBC*> bcs,
                                const MeshFunction<unsigned int>* cell_domains,
                                const MeshFunction<unsigned int>* exterior_facet_domains,
                                const MeshFunction<unsigned int>* interior_facet_domains,
                                bool reset_sparsity,
                                bool add_values,
                                bool finalize_tensor,
                                bool keep_diagonal)
{
  SymmetricAssembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(As, An, a, bcs, bcs,
                     cell_domains, exterior_facet_domains, interior_facet_domains);
}
//-----------------------------------------------------------------------------
void dolfin::symmetric_assemble(GenericMatrix& As,
                                GenericMatrix& An,
                                const Form& a,
                                const std::vector<const DirichletBC*> row_bcs,
                                const std::vector<const DirichletBC*> col_bcs,
                                const MeshFunction<unsigned int>* cell_domains,
                                const MeshFunction<unsigned int>* exterior_facet_domains,
                                const MeshFunction<unsigned int>* interior_facet_domains,
                                bool reset_sparsity,
                                bool add_values,
                                bool finalize_tensor,
                                bool keep_diagonal)
{
  SymmetricAssembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  assembler.keep_diagonal = keep_diagonal;
  assembler.assemble(As, An, a, row_bcs, col_bcs, cell_domains,
                     exterior_facet_domains, interior_facet_domains);
}
//-----------------------------------------------------------------------------
double dolfin::assemble(const Form& a,
                        bool reset_sparsity,
                        bool add_values,
                        bool finalize_tensor)
{
  if (a.rank() != 0)
  {
    dolfin_error("assemble.cpp",
                 "assemble form",
                 "Expecting a scalar form but rank is %d",
                 a.rank());
  }
  Assembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;

  Scalar s;
  assembler.assemble(s, a);
  return s;
}
//-----------------------------------------------------------------------------
double dolfin::assemble(const Form& a,
                        const SubDomain& sub_domain,
                        bool reset_sparsity,
                        bool add_values,
                        bool finalize_tensor)
{
  if (a.rank() != 0)
  {
    dolfin_error("assemble.cpp",
                 "assemble form",
                 "Expecting a scalar form but rank is %d",
                 a.rank());
  }
  Assembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;
  Scalar s;
  assembler.assemble(s, a, sub_domain);
  return s;
}
//-----------------------------------------------------------------------------
double dolfin::assemble(const Form& a,
                        const MeshFunction<uint>* cell_domains,
                        const MeshFunction<uint>* exterior_facet_domains,
                        const MeshFunction<uint>* interior_facet_domains,
                        bool reset_sparsity,
                        bool add_values,
                        bool finalize_tensor)
{
  if (a.rank() != 0)
  {
    dolfin_error("assemble.cpp",
                 "assemble form",
                 "Expecting a scalar form but rank is %d",
                 a.rank());
  }

  Assembler assembler;
  assembler.reset_sparsity = reset_sparsity;
  assembler.add_values = add_values;
  assembler.finalize_tensor = finalize_tensor;

  Scalar s;
  assembler.assemble(s, a, cell_domains, exterior_facet_domains,
                      interior_facet_domains);
  return s;
}
//-----------------------------------------------------------------------------
