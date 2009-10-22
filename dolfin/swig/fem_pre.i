/* -*- C -*- */
// Copyright (C) 2009 Anders Logg
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth Wells, 2007-2009.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Johan Hake, 2008-2009.
//
// First added:  2007-08-16
// Last changed: 2009-10-22

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

// Ignore operator= for DirichletBC to avoid warning
%ignore dolfin::DirichletBC::operator=;
