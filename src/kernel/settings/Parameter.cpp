// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin/dolfin_log.h>
#include <dolfin/utils.h>
#include <dolfin/Parameter.h>

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
  // Do nothing
}
//-----------------------------------------------------------------------------
void Parameter::clear()
{
  identifier = "";
  
  val_real       = 0.0;
  val_int        = 0;
  val_bool       = false;
  val_string     = "";
  
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
  // Save the value of the parameter
  switch ( type ) {
  case REAL:
    
    val_real = va_arg(aptr, real);
    break;
    
  case INT:
    
    val_int = va_arg(aptr, int);
    break;
    
  case BOOL:
    
    val_bool = static_cast<bool>(va_arg(aptr, int));
    break;
    
  case STRING:
    
    // Get the string
    val_string = va_arg(aptr, char *);
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
  real   *p_real;
  int    *p_int;
  int    *p_bool;
  string *p_string;
  
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
    
    p_bool = va_arg(aptr,int *);
    *p_bool = static_cast<int>(val_bool);
    break;
    
  case STRING:
    
    p_string = va_arg(aptr,string *);
    *p_string = val_string;
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
Parameter::operator unsigned int() const
{
  if ( type != INT )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <int>.",
		  identifier.c_str());
  
  if ( val_int < 0 )
    dolfin_error("Assignment not possible. Value is negative and variable is of type <unsigned int>.");

  int val_unsigned_int = val_int;

  return val_unsigned_int;
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
/*Parameter::operator const char*() const
{
  if ( type != STRING )
    dolfin_error1("Assignment not possible. Parameter \"%s\" is not of type <string>.",
		  identifier.c_str());
  
  return val_string.c_str();
}*/
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
  default:
    stream << "[ Parameter of unknown type. ]";
  }
  
  return stream;
}
//-----------------------------------------------------------------------------
