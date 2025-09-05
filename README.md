# FITNESS_TRACKER 🏃🏻📊

Develop Python scripts for preprocessing, visualization, and modeling of accelerometer and gyroscope sensor data to build a machine learning model capable of classifying barbell exercises and accurately counting repetitions.


## Acknowledgements

 - [Hoogendoorn, M. and Funk, B., Machine Learning for the Quantified Self](https://github.com/mhoogen/ML4QS/tree/master)


## Appendix

• This project covers the following components:

• Introduction & Goal: Overview of the quantified self approach, use of the MetaMotion sensor, and dataset description.

• Data Preparation: Converting raw sensor data, reading CSV files, splitting datasets, and data cleaning.

• Data Visualization: Plotting time-series signals for exploratory analysis.

• Outlier Detection: Applying methods such as Chauvenet’s criterion and Local Outlier Factor (LOF).

• Feature Engineering: Extracting frequency-domain features, applying low-pass filters, dimensionality reduction (PCA), and clustering techniques.

• Predictive Modeling: Implementing machine learning models including Naive Bayes, SVMs, Random Forest, and Neural Networks.

• Repetition Counting: Designing and integrating a custom algorithm for accurate repetition detection.




## Features

- Utilizes a feedforward neural network model, achieving 100% accuracy on the dataset.
- Includes a custom algorithm to reliably count exercise repetitions.



## Feedback

If you have any feedback, please reach out at asishb704@gmail.com


## Screenshots

![confusion matrix](https://github.com/Asish-baidya29/FITNESS_TRACKER/blob/main/output_100.png)

![learning curve](https://github.com/Asish-baidya29/FITNESS_TRACKER/blob/main/output%20graph.png)

## Folder Structure
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
