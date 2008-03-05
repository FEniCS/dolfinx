/* getopts.c - 
 *
 * Whom: Steve Mertz <steve@dragon-ware.com>
 * Date: 20010610
 * Why: A exercise in converting my C based getopts() into an OO solution.
 *
*/
/*
 * Copyright (c) 2001-2004 Steve Mertz <steve@dragon-ware.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this 
 * list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * 
 * Neither the name of Dragon Ware Computer Services nor the names of its 
 * contributors may be used to endorse or promote products derived from 
 * this software without specific prior written permission. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS 
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 *
*/

#include <iostream>
#include <string>
#include <map>

#include "getopts.h"

/* Options::Options()
 *
 * Whom: Steve Mertz <steve@dragon-ware.com>
 * Date: 20010610
 * Why:  Initialize lastNumberUsed to '-1' for use in our map.
 *
 * Returns: Nothing
*/
Options::Options()
{
	lastNumberUsed = -1;
}

/* void Options::addOption(string shortName, string longName, string description, bool takeArg)
 * 
 * Whom: Steve Mertz <steve@dragon-ware.com>
 * Date: 20010609
 * Why:  Adds an option to the option map
 *
 * Returns: Nothing
*/
void Options::addOption(std::string shortName, std::string longName, std::string description, bool takeArg)
{
	struct option toBeAdded;

	toBeAdded.shortName = shortName;
	toBeAdded.longName = longName;
	toBeAdded.description = description;
	toBeAdded.takesArg = takeArg;
  toBeAdded.isUsed = false;
	
	/* Increment lastNumberUsed because the number of this new struct will become the new lastNumberUsed
		 This works for the first item too since we initialize lastNumberUsed with -1 to start with.  */
  lastNumberUsed++;
	optionList[lastNumberUsed] = toBeAdded;
}

/* bool Options::parse(int argc, char **argv)
 *
 * Whom: Steve Mertz <steve@dragon-ware.com>
 * Date: 20010609
 * Why:  Parses the command line arguments and puts the arguments of the options 
 *       into their struct.  Also sets the isUsed flag if we have parsed that option.
 *
 * Returns: true  - we have parsed something.
 *          false - we have not parsed anything.
*/
bool Options::parse(int argc, char **argv)
{
	bool realOption, parsedData = false;                       // Set to true if we actually parse something useful.
	
	/* First we must scan for the '-h' and '--help' options */	
  for (int count = 1; count < argc; count++)
		{
			if (!strcmp(argv[count], "-h") || !strcmp(argv[count], "--help"))
				showHelp(argv[0]);                       // We found out that the user wants help.  Go show and exit.
		}	
  
	/* 
		What we are doing here, is cycling through all the arguments that were passed in the program (argv)
		and then we cycle through the list of options that we know about trying to match them up.  If we have
		an unknown option or we have a option which requires an argument but there are none, then we 
		showHelp() and exit.  Otherwise we add the argument to the list and set isUsed to true for later use 
		in cycle().
	*/
	for (int argCount = 1; argCount < argc; argCount++)
		{
			realOption = false;
			for (int listCount = 0; listCount <= lastNumberUsed; listCount++)
				{
					struct option tester = optionList[listCount];
					if ((!tester.shortName.empty() && !strcmp(tester.shortName.c_str(), argv[argCount]+1)) ||
						(!tester.longName.empty() && !strcmp(tester.longName.c_str(), argv[argCount]+2)))
						{
							realOption = true;
							parsedData = true;
							tester.isUsed = true;
							if (tester.takesArg)
								{
									argCount++;
									if ((argCount >= argc) || (*argv[argCount] == '-'))
										showHelp(argv[0]);
									tester.optionArgs = argv[argCount];
								}
							optionList[listCount] = tester;
							break;                               // We have a match, no need to continue with the inner loop.
						}
				}
				
			/* We just parsed an option we don't know what it is, so we will just showHelp() instead and exit. */
			if (!realOption)
				showHelp(argv[0]);
		}		
	return parsedData;                             // Have we did we have any data to parse?
}

/* bool Options::beingUsed(int number)
 *
 * Whom: Steve Mertz <steve@dragon-ware.com>
 * Date: 20010610
 * Why:  Returns whether or not 'number' of our map is being used.
 *
 * Returns: false - This item is not being used currently.
 *          true  - This item is being used currently.
*/
bool Options::beingUsed(int number)
{
	return optionList[number].isUsed;
}

/* int Options::cycle()
 * 
 * Whom: Steve Mertz <steve@dragon-ware.com>
 * Date: 20010609
 * Why:  Cycles through the list of options and returns the number of one that
 *       has it's isUsed flag set.  We go through the list backwards using lastNumberUsed
 *       as our start point.
 *
 * Returns: The number of an option that has it's isUsed flag set.
*/
int Options::cycle()
{
	while (lastNumberUsed >= 0)
		{
			if (beingUsed(lastNumberUsed))
				{
					lastNumberUsed--;
					return lastNumberUsed+1;
				}
			lastNumberUsed--;
		}

  return lastNumberUsed;
}

/* string Options::getArgs(int number)
 *
 * Whom: Steve Mertz <steve@dragon-ware.com>
 * Date: 20010610
 * Why:  Returns the arguments that this option was passed on the
 *       commandline.
 *
 * Returns: optionArgs
*/
std::string Options::getArgs(int number)
{
	return optionList[number].optionArgs;
}

/* void Options::showHelp(char *progName)
 *
 * Whom: Steve Mertz <steve@dragon-ware.com>
 * Date: 20010610
 * Why:  Displays the info for the options which are entered from addOption()
 *
 * Notes: This method really needs to be cleaned up probably to replace the
 *        tabs with something more meaningful later on.
 *
 * Returns: Nothing
*/
void Options::showHelp(char *progName)
{
  std::string usageLine;

	std::cout << "Usage: " << progName << " [options [args]]\n" << std::endl;
  std::cout << "  --help,\t-h\t\t\tDisplays this information" << std::endl;

  for (int counter = 0; counter <= lastNumberUsed; counter++)
		{
			if (!optionList[counter].shortName.empty() && !optionList[counter].longName.empty())
				{
					if (optionList[counter].takesArg)
						usageLine = "--" + optionList[counter].longName + ",\t-" + 
							optionList[counter].shortName + " <args>\t\t";
					else
						usageLine = "--" + optionList[counter].longName + ",\t-" + 
							optionList[counter].shortName + "\t\t\t";
				}
			else if (!optionList[counter].longName.empty())
				{
					if (optionList[counter].takesArg)
						usageLine = "--" + optionList[counter].longName + " <args>\t\t\t";
					else
						usageLine = "--" + optionList[counter].longName + "\t\t\t";
				}
			else if (!optionList[counter].shortName.empty())
				{
					if (optionList[counter].takesArg)
						usageLine = "\t\t-" + optionList[counter].shortName + " <args>\t\t";
					else
						usageLine = "\t\t-" + optionList[counter].shortName + "\t\t\t";
				}
			std::cout << "  " << usageLine << optionList[counter].description << std::endl;
		}
  exit(0);
}



