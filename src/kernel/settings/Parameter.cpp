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

}
//-----------------------------------------------------------------------------
void Parameter::clear()
{
  identifier = "";
  
  val_real       = 0.0;
  val_int        = 0;
  val_bool       = 0;
  val_string     = "";
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
  set(identifier, aptr);
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
    
  case BOOL:
    
    val_bool = va_arg(aptr, bool);
    break;
    
  case STRING:
    
    // Get the string
    val_string = va_arg(aptr, char *);
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
  this->identifier = identifier;
  
  // Variable was changed
  _changed = true;
}
//-----------------------------------------------------------------------------
void Parameter::get(va_list aptr)
{
  double     *p_real;
  int        *p_int;
  bool       *p_bool;
  string     *p_string;
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
    
  case BOOL:
    
    p_bool = va_arg(aptr,bool *);
    *p_bool = val_bool;
    break;
    
  case STRING:
    
    p_string = va_arg(aptr,string *);
    *p_string = val_string;
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
    dolfin_error1("Unknown type for parameter \"%s\".", identifier.c_str());
  }
}
//-----------------------------------------------------------------------------
bool Parameter::matches(const char* identifier)
{
  return this->identifier == identifier;
}
//-----------------------------------------------------------------------------
bool Parameter::matches(string identifier)
{
  return this->identifier == identifier;
}
//-----------------------------------------------------------------------------
bool Parameter::changed()
{
  return _changed;
}
//-----------------------------------------------------------------------------
void Parameter::operator= (const Parameter &p)
{
  identifier = p.identifier;
	 
  val_real       = p.val_real;
  val_int        = p.val_int;
  val_bool       = p.val_bool;
  val_string     = p.val_string;
  val_function   = p.val_function;
  val_vfunction  = p.val_vfunction;
  val_bcfunction = p.val_bcfunction;
  
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
		  identifier.c_str());

  return val_real;
}
//-----------------------------------------------------------------------------
Parameter::operator int() const
{
  if ( type != INT )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <int>.",
		  identifier.c_str());

  return val_int;
}
//-----------------------------------------------------------------------------
Parameter::operator bool() const
{
  if ( type != BOOL )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <bool>.",
		  identifier.c_str());

  return val_bool;
}
//-----------------------------------------------------------------------------
Parameter::operator string() const
{
  if ( type != STRING )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <string>.",
		  identifier.c_str());
  
  return val_string;
}
//-----------------------------------------------------------------------------
Parameter::operator const char*() const
{
  if ( type != STRING )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <string>.",
		  identifier.c_str());
  
  return val_string.c_str();
}
//-----------------------------------------------------------------------------
Parameter::operator function() const
{
  if ( type != FUNCTION )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <function>.",
		  identifier.c_str());
  
  return val_function;
}
//-----------------------------------------------------------------------------
Parameter::operator vfunction() const
{
  if ( type != VFUNCTION )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <vfunction>.",
		  identifier.c_str());
  
  return val_vfunction;
}
//-----------------------------------------------------------------------------
Parameter::operator bcfunction() const
{
  if ( type != BCFUNCTION )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <bcfunction>.",
		  identifier.c_str());
  
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
  case Parameter::BOOL:
    stream << "[ Parameter of type <bool> with value " << p.val_bool << ". ]";
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
