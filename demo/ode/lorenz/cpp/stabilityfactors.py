#!/usr/bin/env python

from numpy import *
from numpy.linalg import norm
from scipy.linalg import expm

def getUU(val, t, U) :
  dummy, u = min(zip(abs(t-val), U), key=lambda u:u[0])
  return u

def stabilityfactors(t, U, Jacobian, show_progress=True, plot=True) :
  # Initial data for the dual
  psi = array((1, 0, 0))
  k   = 0.025
  T   = t[len(t)-1]
  tt  = k

  A = [0]; B = [0]
  S =  [0.0]; tS = [0.0]

  lastprogressprint = 0

  if show_progress :
    print "Computing stability factor as function of time"

  while tt < T :
    if show_progress and tt/T > lastprogressprint+0.05 :
      # show simple text based progress bar
      length = 70
      progress = tt/T
      print "-"*length
      print "%d%%" % int(100*progress),
      print "*" * (progress*length-len("%d%%" % progress))
      print "-"*length
      lastprogressprint = progress

    UU = getUU(tt, t, U)

    JT = transpose(Jacobian(UU));
    A.append(JT)

    # Compute new matrix exponential
    E = expm(k*JT)

    # Multiply solution matrices from the right
    B[1:] = [dot(b,E) for b in B[1:]]
    B.append(E)

    # Compute stability factor
    sum = 0.0
    for A_j, B_j in zip(A[1:], B[1:]) :
      phi = B_j*psi
      sum = sum + k * norm(A_j * phi)

    # Save result
    S.append(sum)
    tS.append(tt)

    # Next time step
    tt = tt + k

  if show_progress : print "Done"

  if plot : 
    import pylab
    pylab.figure(1)
    pylab.semilogy(tS, S)
    pylab.xlabel('T')
    pylab.ylabel('S')
    pylab.show()


if __name__ == '__main__' :
  t = fromfile('solution_t.data', sep=" ")
  U = fromfile('solution_u.data', sep=" ")
  U.shape = (len(U)/3, 3)

  def Jacobian(u) :
    # The usual constants
    s = 10.0;
    r = 28.0;
    b = 8.0/3;

    return array([[ -s      , s   , 0     ],
                  [ r - u[2], -1  , -u[0] ],
                  [ u[1]    , u[0], -b    ]])



  stabilityfactors(t, U, Jacobian)
