% Copyright (C) 2005 Johan Jansson and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% First added:  
% Last changed: 2005

% Load data
primal

% Get minimum and maximum for u and v
m = size(u, 2);
ntot = size(u, 1);
n = ntot / 2 - 1;
h = 1 / n;

x = 0:h:1;

umin = 1e10; umax = -1e10;
vmin = 1e10; vmax = -1e10;
rmin = 1e10; rmax = -1e10;
kmin = 1e10; kmax = -1e10;
for i = 1:m
  umin = min(umin, min(u(1:n + 1, i))); umax = max(umax, max(u(1:n + 1, i)));
  vmin = min(vmin, min(u(n + 2:ntot, i))); vmax = max(vmax, max(u(n + 2:ntot, i)));
  rmin = min(rmin, min(r(n + 2:ntot, i))); rmax = max(rmax, max(r(n + 2:ntot, i)));
  kmin = min(kmin, min(k(n + 2:ntot, i))); kmax = max(kmax, max(k(n + 2:ntot, i)));
end



% Plot solution
%clf
for i = 1:m
  
kimin = 1e10; kimax = -1e10;
kimin = min(kimin, min(k(n + 2:ntot, i))); kimax = max(kimax, max(k(n + 2:ntot, i)));
format long e
disp('Some statistics:');
kimin
kimax
disp('');

  subplot(2, 2, 1)
  plot(x, u(1:n + 1, i))
  axis([0 1 umin umax])
  xlabel('x')
  ylabel('u')

  subplot(2, 2, 2)
  plot(x, u(n + 2:ntot, i))
  axis([0 1 vmin vmax])
  xlabel('x')
  ylabel('v')

  subplot(2, 2, 3)
  plot(x, r(n + 2:ntot, i))
  axis([0 1 rmin rmax])
  xlabel('x')
  ylabel('r')

  subplot(2, 2, 4)
  plot(x, k(n + 2:ntot, i))
  axis([0 1 kmin kmax])
  xlabel('x')
  ylabel('k')
  
  disp('Press any key to continue')
  pause
    
end
