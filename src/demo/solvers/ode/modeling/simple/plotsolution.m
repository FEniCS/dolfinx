% Copyright (C) 2004 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.

primal

clf

subplot(2,1,1)
plot(t,u(1,:))
ylabel('u1')

subplot(2,1,2)
plot(t,u(2,:))
xlabel('t')
ylabel('u2')
