# Bayesian Analysis

This is a repository for bayesian analysis. This framework is built upon the [hic-param-est package](https://github.com/jbernhard/hic-param-est) from [Jonah Bernhard](https://github.com/jbernhard).


Installation
------------

1. Install python3, with packages `emcee`, `h5py`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `corner`.  Use pip to install them if needed

2. If you don't have R, download R from [here](https://cran.cnr.berkeley.edu/)

3. Open an R Console instance by opening the R app or by typing R in the command line.

4. In the R console, type the command `install.packages('lhs')` and pick an appropriate download mirror if prompted. To ensure the package was properly installed, type `library(lhs)` in the R console. If that command runs without error, the package is installed. Close the R console by typing `quit()`.


Test Run with a toy model
-------------------------
A toy model example can be run under the `toy_model/` folder

1. The toy model

	y1 = A exp(-B x^2);
	y2 = C cosh(D x)
	
	A, B, C, D are the model parameters to fit.

2. The prior parameter range is defined in `toy_model/ABCD.txt`

3. Generating the training data sets

	```
	python3 -m src.design -par toy_model/ABCD.txt -n 500 toy_model
	cd toy_model
	for i in `ls main`; do python3 toy_model.py -i main/$i; done
	mkdir model_results; mv run_* model_results
	```
4. Generate the pseudo experimental data

	`cd toy_model; python3 toy_model.py --exp`

5. Run Bayesian Analysis

	The Bayesian analysis can be performed using the Jupyter Notebook `Run_BayesianAnalysis.ipynb`



