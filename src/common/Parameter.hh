// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARAMETER_HH
#define __PARAMETER_HH

#include <iostream>
#include <string.h>

#include <kw_constants.h>
#include <utils.h>
#include <Display.hh>

#define PARAMETER_IDENTIFIER_LENGTH 128

// A small class for internal use in Settings
class Parameter{

public:
  
  Parameter(){

	 sprintf(identifier,"%s","");
	 
	 val_real     = 0.0;
	 val_double   = 0.0;
	 val_int      = 0;
	 val_string   = 0;
	 
	 changed = false;
	 
	 type = type_none;
	 
  }

  ~Parameter(){

	 if ( val_string )
		delete [] val_string;
	 val_string = 0;

  }
  
  void Set(const char *identifier, Type type, va_list aptr){
	 this->type = type;
	 Set(identifier,aptr);
	 changed = false;
  }
  
  void Set(const char *identifier, va_list aptr){
	 
	 char *string;
	 int length;
	 
	 // Save the value of the parameter
	 switch (type){
	 case type_real:
		val_real = va_arg(aptr,real);
		break;
	 case type_double:
		val_double = va_arg(aptr,double);
		break;
	 case type_int:
		val_int = va_arg(aptr,int);
		break;
	 case type_string:
		// Get the string
		string = va_arg(aptr,char *);
		if ( !string ){
		  length = 1;
		  if ( val_string )
			 delete val_string;
		  val_string = new char[length];
		  // Save the string value
		  sprintf(val_string,"");
		}
		else{
		  // Check the length of the string and allocate just enough space
		  for (length=0;string[length];length++);
		  if ( val_string )
			 delete [] val_string;
		  val_string = new char[length+1];
		  // Save the string value
		  sprintf(val_string,"%s",string);
		}
		break;
	 default:
		display->InternalError("Parameter::Set()","Unknown parameter type: %d",type);
	 }

	 // Save the identifier
	 sprintf(this->identifier,"%s",identifier);

	 // Variable was changed
	 changed = true;
	 
  }
  
  void Get(va_list aptr){
	 	 
	 double *p_real;
	 double *p_double;
	 int    *p_int;
	 char   *p_string;
	 
	 // Set the value of the parameter
	 switch (type){
	 case type_real:
		display->Message(10,"Fetching parameter: <%s> = %f",identifier,val_real);
		p_real = va_arg(aptr,real *);
		*p_real = val_real;
		break;
	 case type_double:
		display->Message(10,"Fetching parameter: <%s> = %f",identifier,val_double);
		p_double = va_arg(aptr,double *);
		*p_double = val_double;
		break;
	 case type_int:
		display->Message(10,"Fetching parameter: <%s> = %d",identifier,val_int);
		p_int = va_arg(aptr,int *);
		*p_int = val_int;
		break;	 
	 case type_string:
		display->Message(10,"Fetching parameter: <%s> = \"%s\"",identifier,val_string);
	 	p_string = va_arg(aptr,char *);
		if ( !p_string ){
		  display->InternalError("Parameter:Get()","Unable to write parameter to null pointer.");
		  return;
		}
		sprintf(p_string,"%s",val_string);
		break;
	 default:
		display->InternalError("Parameter::Get()","Unknown parameter type: ",type);
	 }

  }
	 
  bool Matches(const char *string){
	 return ( strcasecmp(string,identifier) == 0 );
  }

  bool Changed(){
	 return changed;
  }
  
  Type TypeOfData(){
	 return type;
  }

  int StringLength(){
	 int i;
	 for (i=0;identifier[i];i++);
	 return i;
  }

  void WriteToFile(FILE *fp, int padding){

	 char pad[PARAMETER_IDENTIFIER_LENGTH];
	 int i=0;
	 for (i=0;i<padding;i++)
		pad[i] = ' ';
	 pad[i] = '\0';

	 char status[PARAMETER_IDENTIFIER_LENGTH];
	 if ( changed )
		sprintf(status,"");
	 else
		sprintf(status," [default]");
	 
	 switch (type){
	 case type_real:
		fprintf(fp,"real <%s>%s = %.16e%s\n",  identifier,pad,val_real,status);
		break;
	 case type_double:
		fprintf(fp,"double <%s>%s = %.16e%s\n",  identifier,pad,val_double,status);
		break;
	 case type_int:
		fprintf(fp,"int    <%s>%s = %d%s\n",     identifier,pad,val_int,status);
		break;
	 case type_string:
		fprintf(fp,"string <%s>%s = \"%s%s\"\n", identifier,pad,val_string,status);
		break;
	 default:
		display->InternalError("Parameter::Set()","Unknown parameter type: ",type);
	 }

  }
  
  void Display(){
	 cout << "<" << identifier << "> = ";
	 switch (type){
	 case type_real:
		cout << val_real << endl;
		break;
	 case type_double:
		cout << val_double << endl;
		break;
	 case type_int:
		cout << val_int << endl;
		break;
	 case type_string:
		cout << "\"" << val_string << "\"" << endl;
		break;
	 default:
		display->InternalError("Parameter::Set()","Unknown parameter type: ",type);
	 }
  }
  
private:

  // A description of the parameter
  char identifier[PARAMETER_IDENTIFIER_LENGTH];

  // Values
  double val_real;
  double val_double;
  int    val_int;
  char  *val_string;

  // True iff the default value was changed
  bool changed;
  
  // Type of data
  Type type;
  
};

#endif
