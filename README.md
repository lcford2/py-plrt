# rtree-lr-combo

This package combines two widely used and interpretable machine learning algorithms: Decision Trees and Linear Regression.
Decision Trees are non-parametric, supervised learning methods that split the data based at a specific threshold for a specific feature in order to optimize a loss function. 
Linear regression is a parametric, supervised learning method that use a linear combination of a set of features to predict a response variable. 

Decision Trees typically predict the mean of the response variable in a particular leaf.
Models that predict the mean are often referred to as the "Null Model" for a particular dataset. 
Replacing the mean prediction with a linear regression formulation for each leaf helps capture linear dependencies between features and the response, thus improving the accuracy of the model.
This also ensures that the final model is a "white-box" because the steps taken to arrive at a solution are available and interpretable. 
Combining a decision tree on top of several linear regression formulations and fitting them together results in data being separated into groups that can be described by linear relationships the best, thus acting as a faux classification layer. 

Portions of this package are inspired by the [DecisionTreeRegressor from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) and the [Regression Tree in Python From Scratch](https://towardsdatascience.com/regression-tree-in-python-from-scratch-9b7b64c815e3) article by [Eligijus Bujokas](https://eligijus-bujokas.medium.com/). 
Visualization methods are adapted from [treelib](https://treelib.readthedocs.io/en/latest/). 
