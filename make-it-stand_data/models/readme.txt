------------- Make It Stand: Balancing Shapes for 3D Fabrication -------------

Romain Prévost (Interactive Geometry Lab, ETHZ Zürich)
Emily Whiting (Interactive Geometry Lab, ETHZ Zürich)
Sylvain Lefebvre (INRIA Nancy)
Olga Sorkine-Hornung (Interactive Geometry Lab, ETHZ Zürich)

ACM SIGGRAPH 2013

http://igl.ethz.ch/projects/make-it-stand/

------------------------------------------------------------------------------

This archive contains 6 models that have been balanced using our algorithm.

Each folder contains 4 different STL files:

- *_inner.stl and *_outer.stl which are the output of our optimization
  representing the inner and outer surface of the carved model

- *_top.stl and *_bottom.stl which are ready to print. They were created
  manually by merging the two previous files and cutting the models into
  two pieces to be able to access the hole inside and remove the support
  material after printing.

The horse model is an exception. There is only one file since we did not
carve inside the model. No manual processing was needed; we directly use the
outer surface after the optimization.

Some models are really delicate to balance in particular the horse standing
on its two legs. So you need a good printer and a perfectly flat surface.

For more details, take a look at our paper, talk slides and video.

------------------------------------------------------------------------------