#!/bin/sh

echo Reconfigurating:
echo

rm -f config.cache
rm -f config.log
rm -f config.status

echo " - Running aclocal"
aclocal
echo " - Running autoconf"
autoconf
echo " - Running automake"
automake -a

echo
