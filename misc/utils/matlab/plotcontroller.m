% Copyright (C) 2005 Anders Logg.
% Licensed under the GNU LGPL Version 2.1.
% 
% Simple script for evaluating the performance of
% different time step controllers.

clf

subplot(2,1,1)
plot(t, k)

subplot(2,1,2)
e = k.*abs(r);
semilogy(t, e);
hold on
plot(t, 2.0*tol*ones(size(t)))
plot(t, 1.0*tol*ones(size(t)))
plot(t, 0.5*tol*ones(size(t)))
grid on
xlabel('t')
ylabel('e')
