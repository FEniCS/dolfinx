// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/utils.h>
#include <dolfin/Vector.h>
#include <dolfin/Parameter.h>

namespace dolfin {
  
  // Default function
  real function_zero (real x, real y, real z, real t)
  {
    return 0.0;
  }

  // Default vector function
  real vfunction_zero (real x, real y, real z, real t, int i)
  { 
    return 0.0;
  }

  // Default boundary condition function
  void bcfunction_zero (BoundaryCondition& bc)
  {
    return;
  }

}

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameter::Parameter()
{
  clear();
}
//-----------------------------------------------------------------------------
Parameter::Parameter(Type type, const char *identifier, va_list aptr)
{
  clear();
  set(type, identifier, aptr);
}
//-----------------------------------------------------------------------------
Parameter::~Parameter()
{
  if ( val_string )
	 delete [] val_string;
  val_string = 0;
}
//-----------------------------------------------------------------------------
void Parameter::clear()
{
  sprintf(identifier,"%s","");
  
  val_real       = 0.0;
  val_int        = 0;
  val_string     = 0;
  val_function   = 0;
  val_vfunction  = 0;
  val_bcfunction = 0;
  
  _changed = false;
  
  type = NONE;
}
//-----------------------------------------------------------------------------
void Parameter::set(Type type, const char *identifier, va_list aptr)
{
  this->type = type;
  set(identifier,aptr);
  _changed = false;
}
//-----------------------------------------------------------------------------
void Parameter::set(const char *identifier, va_list aptr)
{
  char *string;
  int n;
  
  // Save the value of the parameter
  switch ( type ) {
  case REAL:
    
    val_real = va_arg(aptr, real);
    break;
    
  case INT:
    
    val_int = va_arg(aptr, int);
    break;
    
  case STRING:
    
    // Get the string
    string = va_arg(aptr, char *);
    if ( !string ){
      n = 1;
      if ( val_string )
	delete val_string;
      val_string = new char[n];
      // Save the string value
      sprintf(val_string, "%s", "");
    }
    else{
      // Check the length of the string and allocate just enough space
      n = length(string);
      if ( val_string )
	delete [] val_string;
      val_string = new char[n+1];
      // Save the string value
      sprintf(val_string, "%s", string);
    }
    
    break;
    
  case FUNCTION:
    
    val_function = va_arg(aptr, function);
    if ( val_function == 0 )
      val_function = function_zero;

    break;
    
  case VFUNCTION:
    
    val_vfunction = va_arg(aptr, vfunction);
    if ( val_vfunction == 0 )
      val_vfunction = vfunction_zero;

    break;
    
  case BCFUNCTION:

    val_bcfunction = va_arg(aptr, bcfunction);
    if ( val_bcfunction == 0 )
      val_bcfunction = bcfunction_zero;

    break;
    
  default:
    dolfin_error1("Unknown type for parameter \"%s\".", identifier);
  }
  
  // Save the identifier
  sprintf(this->identifier,"%s",identifier);
  
  // Variable was changed
  _changed = true;
}
//-----------------------------------------------------------------------------
void Parameter::get(va_list aptr)
{
  double     *p_real;
  int        *p_int;
  char       *p_string;
  function   *p_function;
  vfunction  *p_vfunction;
  bcfunction *p_bcfunction;
  
  // Set the value of the parameter
  switch ( type ){
  case REAL:
    
    p_real = va_arg(aptr,real *);
    *p_real = val_real;
    break;
    
  case INT:
    
    p_int = va_arg(aptr,int *);
    *p_int = val_int;
    break;
    
  case STRING:
    
    p_string = va_arg(aptr,char *);
    if ( !p_string ){
      dolfin_error("Cannot write parameter value to supplied null pointer.");
      return;
    }
    sprintf(p_string,"%s",val_string);
    break;

  case FUNCTION:
    
    p_function = va_arg(aptr,function *);
    *p_function = val_function;
    break;
    
  case VFUNCTION:

    p_vfunction = va_arg(aptr,vfunction *);
    *p_vfunction = val_vfunction;
    break;
    
  case BCFUNCTION:

    p_bcfunction = va_arg(aptr,bcfunction *);
    *p_bcfunction = val_bcfunction;
    break;
    
  default:
    dolfin_error1("Unknown type for parameter \"%s\".", identifier);
  }
}
//-----------------------------------------------------------------------------
bool Parameter::matches(const char *string)
{
  return strcasecmp(string,identifier) == 0;
}
//-----------------------------------------------------------------------------
bool Parameter::changed()
{
  return _changed;
}
//-----------------------------------------------------------------------------
void Parameter::operator= (const Parameter &p)
{
  sprintf(identifier, "%s", p.identifier);
	 
  val_real       = p.val_real;
  val_int        = p.val_int;
  val_function   = p.val_function;
  val_vfunction  = p.val_vfunction;
  val_bcfunction = p.val_bcfunction;
  
  if ( val_string )
    delete [] val_string;
  val_string = 0;
  
  if ( p.val_string ) {
    val_string = new char[length(p.val_string)];
    sprintf(val_string, "%s", p.val_string);
  }
  
  _changed = p._changed;
  type     = p.type;
}
//-----------------------------------------------------------------------------
void Parameter::operator= (int zero)
{
  if ( zero != 0 )
    dolfin_error("Assignment to int must be zero.");
  
  clear();
}
//-----------------------------------------------------------------------------
bool Parameter::operator! () const
{
  return type == NONE;
}
//-----------------------------------------------------------------------------
Parameter::operator real() const
{
  if ( type != REAL )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <real>.", 
		 identifier);

  return val_real;
}
//-----------------------------------------------------------------------------
Parameter::operator int() const
{
  if ( type != INT )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <int>.",
		  identifier);

  return val_int;
}
//-----------------------------------------------------------------------------
Parameter::operator char*() const
{
  if ( type != REAL )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <string>.",
		  identifier);
  
  return val_string;
}
//-----------------------------------------------------------------------------
Parameter::operator function() const
{
  if ( type != FUNCTION )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <function>.",
		  identifier);
  
  return val_function;
}
//-----------------------------------------------------------------------------
Parameter::operator vfunction() const
{
  if ( type != VFUNCTION )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <vfunction>.",
		  identifier);
  
  return val_vfunction;
}
//-----------------------------------------------------------------------------
Parameter::operator bcfunction() const
{
  if ( type != BCFUNCTION )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <bcfunction>.",
		  identifier);
  
  return val_bcfunction;
}
//-----------------------------------------------------------------------------
// Output
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const Parameter& p)
{
  switch ( p.type ) {
  case Parameter::REAL:
    stream << "[ Parameter of type <real> with value " << p.val_real << ". ]";
    break;
  case Parameter::INT:
    stream << "[ Parameter of type <int> with value " << p.val_int << ". ]";
    break;
  case Parameter::STRING:
    stream << "[ Parameter of type <string> with value " << p.val_string << ". ]";
    break;
  case Parameter::FUNCTION:
    stream << "[ Parameter of type <function>.]";
    break;
  case Parameter::VFUNCTION:
    stream << "[ Parameter of type <vfunction>.]";
    break;
  case Parameter::BCFUNCTION:
    stream << "[ Parameter of type <bcfunction>.]";
    break;
  default:
    stream << "[ Parameter of unknown type. ]";
  }
  
  return stream;
}
//-----------------------------------------------------------------------------
