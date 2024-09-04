# Particle Swarm Optimization - Nature's Intelligence(link to Kaggle notebooks at last)

## Particle Swarm Optimization (PSO) is a population-based optimization algorithm inspired by the social behavior of birds flocking or fish schooling. It's used to solve complex optimization problems by iteratively improving candidate solutions with respect to a given objective function.



### Key Concepts in PSO


**Swarm (Population):**
PSO involves a group of individuals, called particles, that collectively form a swarm.
Each particle represents a potential solution to the optimization problem.
The swarm is initialized with random positions and velocities in the search space.

**Particles:**
Each particle has a position and velocity.
The position corresponds to a candidate solution.
The velocity is the direction and speed with which the particle moves through the search space.

**Objective Function:**
Particles evaluate their fitness based on an objective function, which the algorithm seeks to minimize or maximize.
The fitness of a particle is a measure of how good its current position is as a solution to the problem.

**Personal and Global Bests:**
Each particle keeps track of its personal best position (denoted as p_best), the best solution it has found so far.
The swarm also tracks the global best position (denoted as g_best), which is the best solution found by any particle in the swarm.

**Velocity and Position Update Rules:**
At each iteration, particles adjust their velocities and positions based on the following factors:
Inertia
Cognitive Component(move towards Personal Best)
Social Component (move towards the Global Best)

**Termination Criteria:**
The algorithm iterates until a stopping condition is met, such as a maximum number of iterations or convergence to an acceptable solution.


## The DataSet

### The dataset is the Cleveland Heart Disease dataset taken from the UCI repository. The dataset consists of 303 individuals’ data. There are 14 columns in the dataset(which have been extracted from a larger set of 75). No missing values. The classification task is to predict whether an individual is suffering from heart disease or not. (0: absence, 1: presence)

Original data: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Content
This database contains 13 attributes and a target variable. It has 8 nominal values and 5 numeric values. The detailed description of all these features are as follows:

Age:\
Patients Age in years (Numeric)

Sex:\
Gender (Male : 1; Female : 0) (Nominal) 

cp:\
Type of chest pain experienced by patient. This term categorized into 4 category.\
0 typical angina, 1 atypical angina, 2 non- anginal pain, 3 asymptomatic (Nominal)

trestbps:\
patient's level of blood pressure at resting mode in mm/HG (Numerical)

chol:\
Serum cholesterol in mg/dl (Numeric)

fbs:\
Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false (Nominal)

restecg:\
Result of electrocardiogram while at rest are represented in 3 distinct values\
0 : Normal\
1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of >
0.05 mV) \
2: showing probable or definite left ventricular hypertrophyby Estes' criteria (Nominal)

thalach:\
Maximum heart rate achieved (Numeric)

exang:\
Angina induced by exercise 0 depicting NO 1 depicting Yes (Nominal)

oldpeak:\
Exercise induced ST-depression in relative with the state of rest (Numeric)

slope:\
ST segment measured in terms of slope during peak exercise\
0: up sloping; 1: flat; 2: down sloping(Nominal)

ca:\
The number of major vessels (0–3)(nominal)

thal:\
A blood disorder called thalassemia\
0: NULL\
1: normal blood flow\
2: fixed defect (no blood flow in some part of the heart)\
3: reversible defect (a blood flow is observed but it is not normal(nominal)

target:\
It is the target variable which we have to predict.\
1 means patient is suffering from heart disease and 0 means patient is normal.

## Problem Statement

### Many heart diseases occur due to a variety of reasons, and while in order to detect them, it may become necessary to identify which tests are generally to be done, as it may not be feasible for people to generate so many reports Eg: Our dataset contains 13 features, and patient cannot be expected to have the results of all 13 with them.

**In this case, we use PSO's optimization for the Feature Selection Process**

I have used various classification algorithms such as Logistic Regression, SVC, Random Forest, Gradient Boosting as the objective function for the PSO algorithm, and train the models on the training set, to predict the onset of a Heart disease.

In this case, the particles are different combinations of features in the dataset.

Each particle is represented with a **Binary Vector** of 13 values(0: not selected and 1: selected).

**The algorithm then runs as per given swarm size and iterations and tries to find a combination of selected features which can best predict the onset of heart disease.**

## Solution:

### Process:
I have used 4 different classification algorithms:

Logistic Regression\
Support Vector Classifier\
Random Forest\
Gradient Boosting

These act as an objective target for the particles to reach.\
Additionally, a penalty can be added to each step for the particle as per the achieved accuracy scores.

### Working:

The function returns a subset of the features selected, which are best suited according to the classification algorithm for prediction of heart disease(1 in target variable).

Then, we use all 4 classification algorithms and performance metrics in order to check the accuracy scores.


### The 4 notebooks show the outputs of the classification algorithms.

[Logistic Regression](https://www.kaggle.com/code/jenilkumbhani/pso-lr)\
[SVC](https://www.kaggle.com/code/jenilkumbhani/pso-svc)\
[Random Forest](https://www.kaggle.com/code/jenilkumbhani/pso-rf)\
[Gradient Boosting](https://www.kaggle.com/code/jenilkumbhani/pso-gb)
