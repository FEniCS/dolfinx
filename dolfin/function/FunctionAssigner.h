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
// First added:  2013-09-20
// Last changed: 2013-09-23

#ifndef __DOLFIN_FUNCTION_ASSIGNER_H
#define __DOLFIN_FUNCTION_ASSIGNER_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace dolfin
{

  class Function;
  class FunctionSpace;

  /// This class facilitate assignments between Function and sub
  /// Functions. It builds and caches maps between compatible
  /// dofs. These maps are used in the assignment methods which
  /// perform the actuall assignment. Optionally can a MeshFunction be
  /// passed together with a label, facilitating FunctionAssignment
  /// over sub domains.
  class FunctionAssigner 
  {
  public:

    /// Create a FunctionAssigner between two equally sized functions
    ///
    /// *Arguments*
    ///     assigning_space (_FunctionSpace_)
    ///         The function space of the assigning function
    ///     receiving_space (_FunctionSpace_)
    ///         The function space of the receiving function
    FunctionAssigner(boost::shared_ptr<const FunctionSpace> assigning_space, 
		     boost::shared_ptr<const FunctionSpace> receiving_space);

    /// Create a FunctionAssigner between one mixed Function and
    /// several smaller functions. The smaller components need to add
    /// up to the mixed function.
    ///
    /// *Arguments*
    ///     assigning_space (_FunctionSpace_)
    ///         The assigning function space
    ///     receiving_spaces (std::vector<_FunctionSpace_>)
    ///         The recieving function spaces
    FunctionAssigner(boost::shared_ptr<const FunctionSpace> assigning_space, 
		     std::vector<boost::shared_ptr<const FunctionSpace> > receiving_spaces);

    /// Create a FunctionAssigner between several smaller functions
    /// and one mixed function. The smaller components need to add up
    /// to the mixed function.
    ///
    /// *Arguments*
    ///     assigning_space (_FunctionSpace_)
    ///         The assigning function spaces
    ///     receiving_spaces (std::vector<_FunctionSpace_>)
    ///         The recieving function space
    FunctionAssigner(std::vector<boost::shared_ptr<const FunctionSpace> > assigning_spaces, 
		     boost::shared_ptr<const FunctionSpace> receiving_space);

    void assign(boost::shared_ptr<const Function> assigning_func, 
		boost::shared_ptr<Function> receiving_func) const;

    void assign(std::vector<boost::shared_ptr<const Function> > assigning_funcs, 
		boost::shared_ptr<Function> receiving_func) const;

    void assign(boost::shared_ptr<const Function> assigning_funcs, 
		std::vector<boost::shared_ptr<Function> > receiving_func) const;
    
    /// Destructor
    ~FunctionAssigner();

    /// Return the number of assiging spaces
    inline unsigned int num_assigning_spaces() const
    { return _assigning_spaces.size(); }

    /// Return the number of receiving spaces
    inline unsigned int num_receiving_spaces() const
    { return _receiving_spaces.size(); }

  private:
    
    // Check the compatability of the arguments to the constructor
    void _check_compatability();
    
    // Initialize the transfer indices
    void _init_indices();

    // Shared pointers to the original FunctionSpaces
    std::vector<boost::shared_ptr<const FunctionSpace> > _assigning_spaces;
    std::vector<boost::shared_ptr<const FunctionSpace> > _receiving_spaces;

    // Indices for accessing values from assigning Functions
    std::vector<std::vector<std::size_t> > _assigning_indices;

    // Indices for accessing values to receiving Functions
    std::vector<std::vector<std::size_t> > _receiving_indices;

    // Vector for value transfer between assigning and receiving Function
    std::vector<double> _transfer;

  };
}

#endif
