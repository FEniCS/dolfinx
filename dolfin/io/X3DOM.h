// Copyright (C) 2013 Chris Richardson
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
// First added:  2012-03-05
// Last changed: 2013-05-10

#ifndef __X3DOM_H
#define __X3DOM_H

#include "pugixml.hpp"

namespace dolfin
{

  /// This class implements output of meshes to X3DOM XML 
  /// or HTML or string
  class X3DOM
  {
  public:

    /// Constructor
    explicit X3DOM(const Mesh& mesh);

    /// Destructor
    ~X3DOM();

    std::string write_xml() const;

    std::string write_html() const;

    void save_to_file(const std::string filename);

  private:
  	// XML data
    pugi::xml_document xml_doc;
  };

}

#endif
