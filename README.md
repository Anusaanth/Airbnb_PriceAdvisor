[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/YCTbQ0qx)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10592410&assignment_repo_type=AssignmentRepo)
Project Instructions
==============================

This repo contains the instructions for a machine learning project.

To run project type "cd src" and then"python main.py" in terminal on Visual code studio

## Introduction

The problem I will analyze is how Airbnb's prices change between European countries, as a result of several variables. The dataset I'm utilizing includes information such as price, room type, capacity, super host, ratings, location, and more. I'll be utilizing machine learning techniques to figure out what variables affect Airbnb rental costs the most.

## Objectives

1. Create a machine learning model that can anticipate prices accurately.
2. Decide which features are essential and how they influence the price.
3. Demonstrate to Airbnb hosts how to best price their location.
4. Show local customers and vacationers what qualities they should look for while booking.

## Methodology

To overcome the problem, I intend to use machine learning models such as regression. Regression will be an ideal fit for this dataset because I will be predicting a continuous variable based on several independent variables. I'll begin by exploring the data to have a thorough knowledge of it. Then I'll look for missing or inaccurate values and handle them accordingly. I can use the regression models once the data has been cleaned and prepared. I intend to use linear regression, ridge regression, and random forest regression models. Several metrics, such as mean squared error, mean absolute error, R-squared, and so on, will be used to assess performance. The analytics results can provide valuable information to Airbnb hosts and customers.

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── preprocessing data           <- Scripts to download or generate data and pre-process the data
       │   └── make_dataset.py
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │   └── pre_processing.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       ├── visualization  <- Scripts to create exploratory and results oriented visualizations
       │   └── visualize.py
       │
       └── main.py
