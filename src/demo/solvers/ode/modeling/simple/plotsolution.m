% Copyright (C) 2004-2005 Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% First added:  2004-04-05
% Last changed: 2005

primal

clf

subplot(2,1,1)
plot(t,u(1,:))
ylabel('u1')

subplot(2,1,2)
plot(t,u(2,:))
axis([0 100 -1 1])
xlabel('t')
ylabel('u2')
