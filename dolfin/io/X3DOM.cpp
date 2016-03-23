#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

#include "pugixml.hpp"

#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>

#include "X3DOM.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
X3DOM::X3DOM(const Mesh& mesh) 
{
  // Constructor - Do nothing
}
//-----------------------------------------------------------------------------
X3DOM::~X3DOM()
{
  // Destructor - Do nothing
}
//-----------------------------------------------------------------------------
std::string X3DOM::write_xml() const
{
  return "This will print XML string.";
}
//-----------------------------------------------------------------------------
std::string X3DOM::write_html() const
{
  return "This will print XML string.";
}   
//-----------------------------------------------------------------------------
void X3DOM::save_to_file(const std::string filename)
{
  std::cout<<"This should save to a file"<<std::endl;
}
//-----------------------------------------------------------------------------
