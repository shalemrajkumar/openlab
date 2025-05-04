clear all; close all; clc;

%%% algorithm 
% define function
% sample random points across the input space
% find your objective
% (or)
% set optim = inf or some large value
% and iterate for certain epochs
% find objective ex: sample x randomly and find max(-inf, f(x))

%% approach 1

f = @(x) x.^2;

%generates integers from kill picom

x = randi([-100, 100], 1, 100000) .* rand(1, 100000);

y = f(x);

[out, outi] = min(y);

fprintf('optimal f(%0.5f) = %0.5f\n', x(outi), out);

%% approach 2

optim = Inf;

parfor i = 1:10000

    x = rand(1) .* randi([-100, 100]);
    optim = min(optim, f(x));

end

disp(optim)


% not so useful to find optimum unless:
% 1. we can do for a fixed interval
% 2. you must now if min exist a that interval
% 3. f(x) can be estimated faster


%% plotting

inputs = -100:0.1:100;
outputs = f(inputs);

% plot input vs result

plot(inputs, outputs, 'b-'); hold on;
scatter(x, y, '*');
plot([out, out], [min(y), max(y)], 'r--');
hold off;