y_0 = 2;
[t, y] = ode15s(@(t1, y1)f(t1, y1), 0:0.01:2, y_0);
plot(t,y)
interesting_t = zeros(100);index = 1;
for i = 1:length(y)    
    if y(i) < 1.01 &&  y(i) > 0.99
        interesting_t(index) = t(i);        
        index = index +1;
    end
end
interesting_t;disp([t, y])

function result = f(t, y)
result = -sqrt(y);
end