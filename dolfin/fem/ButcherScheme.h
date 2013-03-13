// Copyright (C) 2013 Johan Hake
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
// First added:  2013-02-15
// Last changed: 2013-03-13

#ifndef __BUTCHERSCHEME_H
#define __BUTCHERSCHEME_H

#include <vector>
#include <boost/shared_ptr.hpp>

#include <dolfin/common/Variable.h>
#include <dolfin/function/FunctionAXPY.h>
#include <dolfin/function/Function.h>

#include "Form.h"

namespace dolfin
{

  /// This class is a place holder for forms and solutions for a multi stage 
  /// Butcher tableau based method

  // Forward declarations
  class Form;
  class Function;
  class DirichletBC;
  class Constant;

  class ButcherScheme : public Variable
  {
  public:

    /// Constructor
    /// FIXME: This constructor is a MESS. Needs clean up...
    ButcherScheme(std::vector<std::vector<boost::shared_ptr<const Form> > > stage_forms, 
		  const FunctionAXPY& last_stage, 
		  std::vector<boost::shared_ptr<Function> > stage_solutions,
		  boost::shared_ptr<Function> u, 
		  boost::shared_ptr<Constant> t, 
		  boost::shared_ptr<Constant> dt,
		  std::vector<double> dt_stage_offset, 
		  unsigned int order,
		  const std::string name,
		  const std::string human_form);

    /// Constructor with Boundary conditions
    ButcherScheme(std::vector<std::vector<boost::shared_ptr<const Form> > > stage_forms, 
		  const FunctionAXPY& last_stage, 
		  std::vector<boost::shared_ptr<Function> > stage_solutions,
		  boost::shared_ptr<Function> u, 
		  boost::shared_ptr<Constant> t, 
		  boost::shared_ptr<Constant> dt, 
		  std::vector<double> dt_stage_offset, 
		  unsigned int order,
		  const std::string name,
		  const std::string human_form,
		  std::vector<const DirichletBC* > bcs);

    /// Return the stages
    std::vector<std::vector<boost::shared_ptr<const Form> > >& stage_forms();

    /// Return the last stage
    FunctionAXPY& last_stage();

    /// Return stage solutions
    std::vector<boost::shared_ptr<Function> >& stage_solutions();
    
    /// Return solution variable
    boost::shared_ptr<Function> solution();

    /// Return solution variable (const version)
    boost::shared_ptr<const Function> solution() const;

    /// Return local time
    boost::shared_ptr<Constant> t();

    /// Return local timestep
    boost::shared_ptr<Constant> dt();

    /// Return local timestep
    const std::vector<double>& dt_stage_offset() const;

    /// Return the order of the scheme
    unsigned int order() const;

    /// Return boundary conditions
    std::vector<const DirichletBC* > bcs() const;

    /// Return true if stage is implicit
    bool implicit(unsigned int stage) const;

    /// Return true if the whole scheme is implicit
    bool implicit() const;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

  private:
    
    // Check sanity of arguments
    void _check_arguments();

    // Vector of forms for the different RK stages
    std::vector<std::vector<boost::shared_ptr<const Form> > > _stage_forms;

    // A linear combination of solutions for the last stage
    FunctionAXPY _last_stage;
    
    // Solutions for the different stages
    std::vector<boost::shared_ptr<Function> > _stage_solutions;

    // The solution
    boost::shared_ptr<Function> _u;

    // The local time 
    boost::shared_ptr<Constant> _t;

    // The local time step
    boost::shared_ptr<Constant> _dt;
    
    // The time step offset. (c from the ButcherTableau)
    std::vector<double> _dt_stage_offset;

    // The order of the scheme
    unsigned int _order;

    // Is the scheme implicit
    bool _implicit;

    // A pretty print representation of the form
    std::string _human_form;

    // The boundary conditions
    std::vector<const DirichletBC* > _bcs;

  };

}

#endif
