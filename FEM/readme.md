Here every .py file contains important functions related to the Finite Element Modelling:

- Elements.py
  - Contains the main Element classes. Only Triangle and Segment are implemented as of right now.
- GaussianQuadratures.py
  - Contains the locations and weights of the Gaussian quadratures, separated into python dictionaries.
- MatrixAssembly.py
  - Contains the functions related to finite element matrices assembly. Most functions take in a Mesh object and geomechanical properties.
- MesherWrapper.py
 - Functions wrapping the use of pygmsh and easing access to important stuff, such as node coordinates and connectivity. Everything is contained into a neat Mesh object with easy matplotlib plotting.
- PoroElasticProperties.py
  - Simple functions linking geomechanical properties.