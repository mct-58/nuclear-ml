# Development Log

All code is my own apart from the plotting, which was AI-assisted.

## Stage 1 - Charged particle in a B field

A proton in a uniform 0.05 T field, integrated with Euler and then with RK4.

Euler gains energy every step, because it uses the acceleration from the start of the step for the whole step. The orbit spirals outwards as a result. Over 7.6 orbits at dt = 1e-9 s, the kinetic energy rose 25.5% and the radius ended up 11% too large. RK4 held the energy constant to about 1 part in 10^12 and matched the Larmor radius (2.09e-7 m) to better than 0.001%.

My first dt (1e-11 s) was so small that Euler looked fine, and I had to increase it before the difference showed at all. Writing RK4 was where I realised position and velocity need to be stepped together as one state.

## Stage 1 - Linear regression

Fitting y = mx + c by gradient descent, with the gradients derived by hand. I got m = 2.626, c = 2.933. The true values were 2.6 and 3, and the least-squares best is 2.628, 2.933.

c converged much more slowly than m. When x isn't centred, the loss surface is a stretched, tilted bowl, and no single learning rate suits both parameters. Centring x and giving each parameter its own learning rate fixed it. I picked the rates by trial and error. The ideal rate for each parameter is 1 divided by the second derivative of the loss with respect to it, which would have converged in one step.

## Stage 2 - Decay chains

U-235 → Th-231 → Pa-231 with real half-lives. I rewrote RK4 as a general function (any state array, any derivative function), so it's no longer tied to position and velocity, and added step-doubling adaptive timestepping.

Bugs: a sign error in the coupling between isotopes (one's loss is the next one's gain), and some thinking about how to store everything. I ended up with one dictionary per isotope.

Th-231's activity matches U-235's within a week, which is secular equilibrium. The more interesting part was the step size. It never went above about 13 days, so 27,000 years took 1.2 million steps. Explicit RK4 is only stable when $\lambda \Delta t \lesssim 2.8$, and for Th-231 that caps dt at about 4 days for the entire run, even long after the thorium has settled. That's what stiffness is, and adaptive stepping doesn't get around it. An implicit method would.

## Stage 2 - Neural network

One hidden layer of 20 ReLU neurons, fitting sin x. The forward pass, backprop and training are all written by hand in NumPy.

Most of the work was getting the gradients right. At first I had the sums in the wrong places: a sum over neurons where there shouldn't be one, and no sum over data points where there should. I also used ReLU instead of its derivative. In the code, v had to be reshaped into a column to broadcast correctly, and the bias gradient had to be reshaped to match b's (n, 1) shape. Without that, the update quietly turns b into an n×n array.

I checked every gradient against central finite differences. The worst relative error was about 1e-9. Final loss was 0.006, against a noise floor of 0.0049.