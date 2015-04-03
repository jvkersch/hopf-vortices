Overview
--------

This package contains cython code implementing a geometric variational integrator for the dynamics of point vortices of the sphere.  The discrete equations of motion live on the three-dimensional sphere/the Lie group SU(2), which allows for techniques from discrete Lagrangian mechanics to be used.  More information can be found in the following article: 

> Joris Vankerschaver, Melvin Leok: 
> _A novel formulation of point vortex dynamics on the sphere: geometrical and numerical aspects_. Journal of Nonlinear Science (2014), Volume 24, Issue 1, pp 1-37
> [arXiv preprint](http://arxiv.org/abs/1211.4560)

How to run this thing
---------------------

First, compile the included Cython files by running 

    cd hopf && make

This has to be done only once.

In order to use this package, your PYTHONPATH has to point to the `hopf` directory. From within the root directory (`hopf_vortices`) of this package, run

    source set_python_path.sh

