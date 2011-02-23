/* -*- C -*- */
// Copyright (C) 2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth Wells, 2007-2009.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2007-08-16
// Last changed: 2011-02-09

// ===========================================================================
// SWIG directives for the DOLFIN fem kernel module (pre)
//
// The directives in this file are applied _before_ the header files of the
// modules has been loaded.
// ===========================================================================

//-----------------------------------------------------------------------------
// To simplify handling of shared_ptr types in PyDOLFIN we ignore the reference
// version of constructors to these types
//-----------------------------------------------------------------------------
%ignore dolfin::EqualityBC::EqualityBC(const FunctionSpace&, uint);
%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&,
					 const GenericFunction&,
					 const SubDomain&,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&,
					 const GenericFunction&,
					 const MeshFunction<uint>&,
					 uint,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&,
					 const GenericFunction&,
					 uint,
					 std::string method="topological");

%ignore dolfin::PeriodicBC::PeriodicBC(const FunctionSpace&, const SubDomain&);

%ignore dolfin::DirichletBC(const FunctionSpace&,
                            const GenericFunction&,
                            const std::vector<std::pair<uint, uint> >&,
                            std::string method="topological");

%ignore dolfin::Form::set_mesh(const Mesh& mesh);

%ignore dolfin::Form::set_coefficient(std::string name,
				      const GenericFunction& coefficient);

%ignore dolfin::Form::set_coefficients(std::map<std::string,
				       const GenericFunction*> coefficients);

//-----------------------------------------------------------------------------
// Ignore operator= for DirichletBC to avoid warning
//-----------------------------------------------------------------------------
%ignore dolfin::DirichletBC::operator=;

//-----------------------------------------------------------------------------
// Ignore one of the constructors for DofMap to avoid warning
//-----------------------------------------------------------------------------
%ignore dolfin::DofMap::DofMap(boost::shared_ptr<const ufc::dofmap>, const Mesh&);

//-----------------------------------------------------------------------------
// Modifying the interface of BoundaryCondition
//-----------------------------------------------------------------------------
%ignore dolfin::BoundaryCondition::function_space;
%rename (_function_space) dolfin::BoundaryCondition::function_space_ptr;

//-----------------------------------------------------------------------------
// Instantiate Hierarchical Form
//-----------------------------------------------------------------------------
namespace dolfin {
  class Form;
  class VariationalProblem;
  class DirichletBC;
}

%template (HierarchicalForm) dolfin::Hierarchical<dolfin::Form>;
%template (HierarchicalVariationalProblem) \
          dolfin::Hierarchical<dolfin::VariationalProblem>;
%template (HierarchicalDirichletBC) dolfin::Hierarchical<dolfin::DirichletBC>;
