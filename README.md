# CSC2515_Fall_2017_Paper "Follow the Trend" Examination of Lattice Regression compared to Lasso and Random Forest
Code and results from CSC2515 Fall 2017 Paper at the University of Toronto by Krist Papadopoulos

Paper includes experiments on the Boston Housing Dataset using the algorithms below to examine whether including linear biases such as prior assumptions of monotonicity between variables in lattice regression can improve performance of predictions by avoiding overfitting to noise in the data:

1) Lasso Regression
2) Random Forest Regression
3) Lattice Regression: Calibrated Linear 1-D Lattice, Calibrated Lattice and Random Tiny Little Lattices

The algorithms performance in terms of mean squared error were evaluated on the dataset with and without outliers.

Lattice Regression algorithms, Calibrated Lattice and Random Tiny Little Lattices (Gupta, 2017) were more effective than lasso regression and random forest in lower mean square error and lower standard deviation of predictions.

Maya Gupta, Jan Pfeifer, Seungil You.  Tensorflow Lattice: Flexibility Empowered by Prior Knowledge.  Google Research Blog, 2017.  URL  https://research.googleblog.com/2017/10/tensorflow-lattice-flexibility.html

