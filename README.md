# Recommender systems

The goal of this repository is to facilitate a quick and easy starting point for recommender system projects by providing a set of notebooks testing various state-of-the-art approaches.

## Input data 
All experiments have been performed on the MovieLens dataset, which can be downloaded [here](https://grouplens.org/datasets/movielens/100k/). However, the relevant files have been pushed to the /data folder in this repository.

## Repository structure

### Naive baselines
- Notebook **NaiveBaselines.ipynb** implements several naive baselines
### Recommnder-system libraries
- [scipy SVD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html) for MF baseline model: **Libraries.ipynb**

- [surprise](http://surpriselib.com/): best performing models are SVD- FUNK, SVD++ and KNN. A usage example is provided in **Libraries.ipynb**

- Incremental learning [CF-step](https://pypi.org/project/cf-step/) and [Flurs](https://github.com/takuti/flurs) 

- [Microsoft's recommenders](https://github.com/microsoft/recommenders)


### NN-based solutions

- Pytorch: **PytorchMF.ipynb** implements simple matrix factorization, **PytorchMFDeeper.ipynb** addes several layers to the MF network and **PytorchMFextraFeatures.ipynb** uses additional user/item features
- Keras, using [TensorFlow Recommenders](https://github.com/tensorflow/recommenders) : **TensorflowRecommenders.ipynb**

## Useful resources

### Papers
[Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
### Blogs
[Pytorch blog](https://blog.fastforwardlabs.com/2018/04/10/pytorch-for-recommenders-101.html), [Overview blog](https://towardsdatascience.com/a-complete-guide-to-recommender-system-tutorial-with-sklearn-surprise-keras-recommender-5e52e8ceace1), [Older list of libraries](https://techairesearch.com/overview-of-matrix-factorization-techniques-using-python/)


[Probabilistic MF](https://github.com/ocontreras309/ML_Notebooks/blob/master/PMF_Recommender_Systems.ipynb) explained in this [blog](https://towardsdatascience.com/pmf-for-recommender-systems-cbaf20f102f0).

### Videos
[SVD](https://www.youtube.com/watch?v=rYz83XPxiZo&t=1s&ab_channel=MITOpenCourseWare)