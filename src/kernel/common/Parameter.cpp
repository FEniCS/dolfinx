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
  if ( val_string )
	 delete [] val_string;
  val_string = 0;
}
//-----------------------------------------------------------------------------
void Parameter::clear()
{
  sprintf(identifier,"%s","");
  
  val_real     = 0.0;
  val_int      = 0;
  val_string   = 0;
  val_function = 0;
  
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
		sprintf(val_string,"");
	 }
	 else{
		// Check the length of the string and allocate just enough space
		n = length(string);
		if ( val_string )
		  delete [] val_string;
		val_string = new char[n+1];
		// Save the string value
		sprintf(val_string,"%s",string);
	 }
	 
	 break;

  case FUNCTION:
	 
	 val_function = va_arg(aptr, function);
	 break;
	 
  default:
	 cout << "Unknown type for parameter \"" << identifier << "\"." << endl;
  }
  
  // Save the identifier
  sprintf(this->identifier,"%s",identifier);
  
  // Variable was changed
  _changed = true;
}
//-----------------------------------------------------------------------------
void Parameter::get(va_list aptr)
{
  double   *p_real;
  int      *p_int;
  char     *p_string;
  function *p_function;
  
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
		cout << "Parameter:Get(): Unable to write parameter to null pointer." << endl;
		return;
	 }
	 sprintf(p_string,"%s",val_string);
	 break;

  case FUNCTION:

	 p_function = va_arg(aptr,function *);
	 *p_function = val_function;
	 break;
	 
  default:
	 cout << "Unknown type for parameter \"" << identifier << "\"." << endl;
  }
}
//-----------------------------------------------------------------------------
bool Parameter::matches(const char *string)
{
  return ( strcasecmp(string,identifier) == 0 );
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
	 
  val_real     = p.val_real;
  val_int      = p.val_int;
  val_function = p.val_function;

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
  // FIXME: Use logging system
  if ( zero != 0 ) {
	 cout << "Assignment to int must be zero." << endl;
	 exit(1);
  }
  clear();
}
//-----------------------------------------------------------------------------
bool Parameter::operator! () const
{
  return type == NONE;
}
//-----------------------------------------------------------------------------
