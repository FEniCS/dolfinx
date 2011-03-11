// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-03-09
// Last changed: 2011-03-11

#ifndef __DOMAIN_ASSIGNER_H
#define __DOMAIN_ASSIGNER_H

#include <boost/shared_ptr.hpp>

namespace dolfin
{

  class Form;
  template <class T> class MeshFunction;

  /// These classes are used for assignment of domains to forms:
  ///
  ///   a.cell_domains = cell_domains
  ///   a.exterior_facet_domains = exterior_facet_domains
  ///   a.interior_facet_domains = interior_facet_domains
  ///
  /// where the arguments can be either objects/references or
  /// shared pointers to MeshFunctions.

  /// Assignment of cell domains
  class CellDomainAssigner
  {
  public:

    // Constructor
    CellDomainAssigner(Form& form) : form(form) {}

    // Assign reference
    const CellDomainAssigner& operator= (const MeshFunction<uint>& domains);

    // Assign shared pointer
    const CellDomainAssigner& operator= (boost::shared_ptr<const MeshFunction<uint> > domains);

  private:

    // The form
    Form& form;

  };

  /// Assignment of exterior facet domains
  class ExteriorFacetDomainAssigner
  {
  public:

    // Constructor
    ExteriorFacetDomainAssigner(Form& form) : form(form) {}

    // Assign reference
    const ExteriorFacetDomainAssigner& operator= (const MeshFunction<uint>& domains);

    // Assign shared pointer
    const ExteriorFacetDomainAssigner& operator= (boost::shared_ptr<const MeshFunction<uint> > domains);

  private:

    // The form
    Form& form;

  };

  /// Assignment of interior facet domains
  class InteriorFacetDomainAssigner
  {
  public:

    // Constructor
    InteriorFacetDomainAssigner(Form& form) : form(form) {}

    // Assign reference
    const InteriorFacetDomainAssigner& operator= (const MeshFunction<uint>& domains);

    // Assign shared pointer
    const InteriorFacetDomainAssigner& operator= (boost::shared_ptr<const MeshFunction<uint> > domains);

  private:

    // The form
    Form& form;

  };

}

#endif
