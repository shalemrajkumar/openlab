clear all; close all; clc;

%% Solver

% question: find the time t at which height of a tank with flow though a pipe given by hdot = -(h)**1/2

[t_out, y_out] = ode15s(@(t, y)fun(t, y), [0, 2], 2);
plot(t_out,y_out)

results = {}; % a cell array init

for time = 1:length(t_out)
    if y_out(time) < 1.1 && y_out(time) > 0.95
        % Append to the cell array {time, y_out}
        results{end+1, 1} = t_out(time);  % Store time in first column in next row
        results{end, 2} = y_out(time);    % Store corresponding y_out in second column
    end
end

disp(results)


%% Function
function height = fun(t,y)
height = -sqrt(y);
end