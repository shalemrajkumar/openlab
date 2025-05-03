### Contents


#### Algorithms

##### Optimization

###### 1D Optimizations

- unimodal functions i.e. single max / min in $[a, b]$ 
- One property of unimodal functions that they increase or decrease a to x* and decrease or increase from x* ti b 

> `Golden section search`

**Algorithm**

*conditions* 

- f(x) is 1D 
- f(x) should have a minima v maxima in $[a, b$
- if its maxima use -f(x) as objective function 

1. choose a and b
2. find golden ratio $/frac{/sqrt{5} - 1}{2}$
3. d = GR * (b - a)
4. find x1 = a + d
5. find x2 = b - d
6. find f(x2) and f(x1)
7. if f(x1) < f(x2) (i.e after u move along a distace GR we see we go close to minima with with x1 than x2)
8. use x2 as left bound (replace a)
9. use x1 as x2 and find new x1 repeat
10. else
11. use x1 as right bound (replace b)
12. use x2 as x1 and find new x2 repeat

> Davis Swann campey algorithm :: Quadratic fit search

*conditions*

- Same as previous method
- Approximating the function at three points as quadratic and finding its minima iteratively


> Neider-Mead Algorithm




#### Tree view of directories
