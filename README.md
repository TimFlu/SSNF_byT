# SSNF_byT
Hi Tim,

As promised, here is the material to start reading when you have some time (there is no rush since we will start on September 25th).
Keep in mind that you don’t have to go through everything, as it is a lot of stuff :)

- Normalizing flows in general:
    - [1] repo containing link to the main resources, regularly updated (most of the stuff I mention in what follows is reported here as well)
    - [2] a presentation that Davide did at an ETH group meeting, with stuff from the papers already pre-digested (quite similar to the one you linked)
    - [3] interesting introductory paper 
    - [4] another interesting interesting introductory paper
    - [5] main paper about Neural Spline Flows, which is the main kind of transform used in what I've done so far
- Flows4Flows (i.e. the method I'm trying to tune to perform the corrections):
    - [6] paper
    - [7] gh repo
- CQR (i.e. the method already in place and that we're trying to "upgrade" using NFs)
    - [8] Higgs -> gamma gamma differential cross sections paper, see 4.2.1
- Code
    - [9] main repo where I've been implementing the procedure (there are multiple branches due to the different attempts - don't worry too much about it now since it will require deeper explanation)
    - [10] package containing the flows we're using now 
    - [11] another package that we might investigate

On top of this, I will soon add to a repo some small prototypes that I have used here and there to study the problem without using the full setup - which are probably the best place where to start.
Of course I will explain most of this in much better detail when we will start.

Cheers,
Massimiliano

[1] https://github.com/janosh/awesome-normalizing-flows <br />
[2] https://indico.cern.ch/event/1249252/contributions/5248861/attachments/2584931/4459247/23_01_20%20-%20MEMFlow%20project.pdf <br />
[3] https://arxiv.org/pdf/1912.02762.pdf <br />
[4] https://arxiv.org/pdf/1908.09257.pdf <br />
[5] https://arxiv.org/pdf/1906.04032.pdf <br />
[6] https://arxiv.org/pdf/2211.02487.pdf <br />
[7] https://github.com/jraine/flows4flows <br />
[8] https://arxiv.org/abs/2208.12279 <br />
[9] https://github.com/maxgalli/ShowerShapesNormalizingFlow <br />
[10] https://github.com/bayesiains/nflows <br />
[11] https://github.com/francois-rozet/zuko.git <br />
