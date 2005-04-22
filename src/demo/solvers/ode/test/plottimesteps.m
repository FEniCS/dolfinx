% Copyright (C) 2005 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.

% Load data
primal

% Compute maximum norm of residual as function of time
rmax = max(abs(r));

% Plot time steps k and product k|R|
clf

subplot(3, 1, 1)
semilogy(t', k')
xlabel('t')
ylabel('k')

subplot(3, 1, 2)
semilogy(t', k' .* abs(r'))
grid on
xlabel('t')
ylabel('kR')

subplot(3, 1, 3)
semilogy(t, max(k .* abs(r)))
grid on
xlabel('t')
ylabel('|kR|')

format short e
disp(['max |kR| = ' num2str(max(max(k .* abs(r))))])
