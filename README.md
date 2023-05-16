# Machine Learning Cookbook
This file summarizes approaches not to forget when implementing an ML application. I will be slowly adding new ideas and approaches as a learn them.

## Checklist
1. Read the assignment in detail.
1. Look online, someone has probably dealt with a similar issue and already knows what works and what does not.
1. Obtain information about the data set, how it was created, is it biased, can i.i.d. be assumed and if not what can.
1. Visualize the data set if possible:
    * image data/2D array data plot as images,
    * plot histograms, scatter plot, cross-correlation heat map, PCA, LDA for tabular data if possible or at least scroll through it, 
    * time series data/sound/other signals plot simply as a function and try converting to frequency with FFT or similar algorithm if possible.
1. Split the data set to train and test sets, 80/20 usually works well, some of the pre-processing steps might precede the split, but make sure that there is **no leakage** of the test set to the train set, i.e. normalize/standardize etc. only after the split with the train statistics.
1. Prepare the data set (tabular data):
    * identify types of variables, i.e. compounded columns (e.g. columns capturing an interval), numerical continuous, numerical discrete, categorical, categorical ordinal, boolean, textual, unique categorical identifiers, percentage columns, ...,
    * in most cases (depending on the ML algorithm, i.e. decision trees or random forests might not need the following steps) split compounded columns to atomic values, one-hot encode categorical and boolean columns, map categorical ordinal columns to some interval 0 to 1 will probably be the best, convert textual data to embeddings or some other encoding (possibly perform some ReGex cleaning, stemming/lemmatization and stop word removal first), drop unique categorical identifiers, convert percentages to an interval between 0 and 1, ...,
    * identify and possibly remove outliers if the algorithm (e.g. mean estimation) cannot deal with them, use:
        * histograms and remove the tails, i.e. 2/3 sigma rule,
        * clustering with K-means, DBSCAN, GMMs and set the threshold based on the assumptions over the data set, e.g. 2 percent are outliers, make sure to use the correct distance or similarity measure (Euclidean distance might not work on multi-dimensional data, cosine similarity might be better),
        * auto-encoders - samples that cannot be well reconstructed might be outliers,
    * identify what are missing values, e.g. empty cells, values out of range, zeros, ...,
    * identify why are the values missing:
        * MCAR - missing values are randomly distributed across all records and do not depend on the recorded or missing values,
        * MAR - missing values depend on the recorded values but not on the missing values and sometimes can be recovered,
        * MNAR - missing data depend on the missing values themselves, e.g. salary is missing because it might be to low/high,
    * impute the missing values:
        * mean, median or a constant for continuous numerical values,
        * median or constant (or mean if the change of data type is not an issue) for discrete numerical values,
        * mode, constant, new category or some fuzzy value (likes, likes a bit, dislikes a bit, dislikes) for categorical columns,
    * normalize/standardized the (numerical) data:
        * usually to zero mean and unit variance,
        * decorrelation/whitening might be useful in some cases,
        * standardization to some interval, e.g from -1.0 to 1.0, might work in other cases,
    * visualize the data set again, check for linear correlation with PCA and LDA (inspect if the data are linearly separable), use other techniques like Kendall rank correlation, Canonical correlation, Distance correlation or Maximal information coefficient for non-linear dependencies.
1. Prepare the data set (image and time series data):
    * try to remove outliers,
    * convert to the desired format,
    * normalize/standardized, e.g pixel values from 0-255 to 0.0-1.0.
1. Create a **baseline** implementation or at least calculate the probability of a random guess being correct.
1. Use some well known ML algorithm for the data set, e.g. SVM, Decision Trees, Random Forests, SIFT for images, etc.
1. Use some pre-trained model if possible, do just transform learning (adjust the number of re-learned layers by the size of the data set), make sure to normalize the inputs to a format expected by the model,
1. Define your own DNN model (probably use Tensorflow for regular architectures and maybe PyTorch for some more custom ones):
    * CNN, auto-encoder, GAN, stable diffusion, ..., for image data,
    * RNN (LSTM, GRU) or transformer with time encoding of the inputs for time series data,
    * transformer with positional encoding of the inputs for textual data (converted to embeddings or with an embedding layer),
    * or start with an MLP as a baseline.
1. Train the model:
    * use a validation set (split 80/20 the train set) or do cross-validation on small data sets, never touch the test set,
    * save the randomly initialized weights,
    * by default try the Adam optimizer with the default learning rate of 0.001, it usually performs the best,
    * set the number of epochs very hight and use early stopping to maximize/minimize some parameter as the stopping criterion,
    * reset the weight to the initialized state and retrain the model on the whole train set for the number of epochs discovered by the early stopping,
    * add regularization to generalize better, usually dropout is the best (L2 and L1 might be useful in some cases),
    * add batch normalization to smoothen the training landscape (batch normalization also introduces noise),
    * specify some learning rate schedule,
    * augment the train data set, e.g rotations, flips, noise on images,
    * modify the DNN architecture (number of layers, types of layers, neurons per layer, etc.),
    * change the random seed,
    * tune other hyper-parameters as much as necessary.
1. Evaluate the best performing model based on the validation runs on the test set, use:
    * accuracy fro single class prediction,
    * MSE for regression,
    * precision, recall and F1 score for multi-class prediction, informational retrieval, ...,
    * BLUE or other specific evaluation techniques for specific tasks.
1. Possibly do perform an ablation study and remove some parts of the model, simpler model with the same performance is better.
1. Possibly create an ensemble of more models to further improve performance.
1. Optimize the model for the intended architecture:
    * usually TFLite model is the best option for phones or possibly use the Android NNAPI (or an equivalent on IOS) to define the model in C++,
    * TFLite is also the baseline for embedded devices if possible, otherwise try to improve the algorithm with available HW support like SIMD instructions,
    * optimize the model for GPU with cuDNN, cuBLAS or write own kernels to run on the GPU in CUDA. 
