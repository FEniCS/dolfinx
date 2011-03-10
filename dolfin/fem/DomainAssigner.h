// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-03-09
// Last changed: 2011-03-10

#ifndef __DOMAIN_ASSIGNER_H
#define __DOMAIN_ASSIGNER_H

#include <boost/shared_ptr.hpp>

namespace dolfin
{

  class Form;
  template <class T> class MeshFunction;

  /// This class is used for assignment of domains to forms:
  ///
  ///   a.cell_domains = cell_domains
  ///   a.exterior_facet_domains = exterior_facet_domains
  ///   a.interior_facet_domains = interior_facet_domains
  ///
  /// where the arguments can be either objects/references or
  /// shared pointers to MeshFunctions.

  class DomainAssigner
  {
  public:

    /// Constructor
    DomainAssigner(Form& form);

    /// Destructor
    virtual ~DomainAssigner();

    /// Assign reference
    const DomainAssigner& operator= (const MeshFunction<uint>& domains);

    /// Assign shared pointer
    const DomainAssigner& operator= (boost::shared_ptr<const MeshFunction<uint> > domains);

  protected:

    // Assignment implemented by subclasses
    virtual void assign(boost::shared_ptr<const MeshFunction<uint> > domains) = 0;

    // The form
    Form& form;

  };

  /// Assignment of cell domains
  class CellDomainAssigner : public DomainAssigner
  {
  public:

    CellDomainAssigner(Form& form) : DomainAssigner(form) {}

  protected:

    void assign(boost::shared_ptr<const MeshFunction<uint> > domains);

  };

  /// Assignment of exterior facet domains
  class ExteriorFacetDomainAssigner : public DomainAssigner
  {
  public:

    ExteriorFacetDomainAssigner(Form& form) : DomainAssigner(form) {}

  protected:

    void assign(boost::shared_ptr<const MeshFunction<uint> > domains);

  };

  /// Assignment of interior facet domains
  class InteriorFacetDomainAssigner : public DomainAssigner
  {
  public:

    InteriorFacetDomainAssigner(Form& form) : DomainAssigner(form) {}

  protected:

    void assign(boost::shared_ptr<const MeshFunction<uint> > domains);

  };

}

#endif
