clear all; close all; clc;

% function
fun = @(x) x.^2;

% defined grid
grid = linspace(-100, 100, 10000);

% evaluating the function at grid assuming our min lies in this interval 
y = fun(grid);

plot(grid, y)

%% method 2

%uniformely sample the input range

%samples = min_r : 0.1 : max_r;

% objective function 

obj = @(x, y)(x-1).^2 + (y-1).^2;

min_r = -5; max_r = 5; step = 0.5;

% generate grid samples

[xg, yg] = meshgrid(min_r:step:max_r, min_r:step:max_r);

size(yg)
xg

