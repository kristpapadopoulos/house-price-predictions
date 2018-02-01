## University of Toronto - Department of Computer Science CSC2515 Fall 2017 Paper "Follow the Trend" an examination of lattice regression compared to lasso and random forest regression by Krist Papadopoulos

This repository includes an archive with the code and paper.

The paper included experiments on the Boston Housing Dataset using the algorithms below to examine whether including linear biases such as prior assumptions of monotonicity between variables in lattice regression can improve performance of predictions by avoiding overfitting noise in the data:

1) Lasso Regression
2) Random Forest Regression
3) Lattice Regression: Calibrated Linear 1-D Lattice, Calibrated Lattice and Random Tiny Little Lattices

The algorithms performance in terms of mean squared error were evaluated on the dataset with and without outliers.

Lattice Regression algorithms, Calibrated Lattice and Random Tiny Little Lattices (Gupta, 2017) were more effective than lasso  random forest regression in lower mean square error and lower standard deviation of predictions.

Maya Gupta, Jan Pfeifer, Seungil You.  Tensorflow Lattice: Flexibility Empowered by Prior Knowledge.  Google Research Blog, 2017.  URL  https://research.googleblog.com/2017/10/tensorflow-lattice-flexibility.html

