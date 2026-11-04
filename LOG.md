# Development Log

## Stage 1 - Charged particle in a B field
**Aim: To simulate how any charged particle (in this case a proton) will move in a magnetic field when given an initial velocity.**\
**Problems and Fixes: Initially the main problem was inaccurate integration for each step, this was due to using euler which causes the system to gain energy with every step error. Instead i implemented an RK4 integrator which worked much better and this was shown on the comparison plots.**
****
****
