#ifndef __FUNCTION_HH
#define __FUNCTION_HH

#include <stdio.h>
#include <string.h>

typedef real (*FunctionPointer)(real,real,real,real);

class Function{
public:

  void Set(const char *name){
	 sprintf(this->name,"%s",name);
	 this->f = 0;
  }
  
  void Set(const char *name, FunctionPointer f){
	 sprintf(this->name,"%s",name);
	 this->f = f;
  }

  bool Matches(const char *name){
	 return ( strcasecmp(this->name,name) == 0 );
  }
  
  char name[DOLFIN_WORDLENGTH];  
  FunctionPointer f;

};

#endif
