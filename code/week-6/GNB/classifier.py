import numpy as np
import random
from math import sqrt, pi, exp

def gaussian_prob(obs, mu, sig):
    # Calculate Gaussian probability given
    # - observation
    # - mean
    # - standard deviation
    num = (obs - mu) ** 2
    denum = 2 * sig ** 2
    norm = 1 / sqrt(2 * pi * sig ** 2)
    return norm * exp(-num / denum)

# Gaussian Naive Bayes class
class GNB():
    # Initialize classification categories
    def __init__(self):
        self.classes = ['left', 'keep', 'right']

    # Given a set of variables, preprocess them for feature engineering.
    def process_vars(self, vars):
        # The following implementation simply extracts the four raw values
        # given by the input data, i.e. s, d, s_dot, and d_dot.
        s, d, s_dot, d_dot = vars
        
        return s, d, s_dot, d_dot

    # Train the GNB using a combination of X and Y, where
    # X denotes the observations (here we have four variables for each) and
    # Y denotes the corresponding labels ("left", "keep", "right").
    def train(self, X, Y):
        '''
        Collect the data and calculate mean and standard variation
        for each class. Record them for later use in prediction.
        '''
        # data = [s, d, s_dot, d_dot] 4개의 float list
        # label = 'left', 'right', 'keep' 3가지
        train_data = {
                      'left' : {'datas' : [[], [], [], []],},
                      'right' : {'datas' : [[], [], [], []],},
                      'keep' : {'datas' : [[], [], [], []],},
                      }

        self.means = {}
        self.stds = {}

        for x, label in zip(X, Y):
            # x = [s, d, s_dot, d_dot]
            for i, data in enumerate(x):
                train_data[label]['datas'][i].append(data)
        
        for class_name in self.classes:
            class_mean = np.mean(train_data[class_name]['datas'], axis=1)   # s, d, s_dot, d_dot 별 평균 계산
            class_std = np.std(train_data[class_name]['datas'],axis=1)      # s, d, s_dot, d_dot 별 분산 계산
            
            self.means[class_name] = class_mean
            self.stds[class_name] = class_std

    # Given an observation (s, s_dot, d, d_dot), predict which behaviour
    # the vehicle is going to take using GNB.
    def predict(self, observation):
        '''
        Calculate Gaussian probability for each variable based on the
        mean and standard deviation calculated in the training process.
        Multiply all the probabilities for variables, and then
        normalize them to get conditional probabilities.
        Return the label for the highest conditional probability.
        '''
        prob_per_class = []
        best_prob = -1
        for class_name in self.classes:
            prob = 1.0
            for i, var in enumerate(observation):
                class_i_mu = self.means[class_name][i]
                class_i_std = self.stds[class_name][i]
                prob *= gaussian_prob(var, class_i_mu, class_i_std)

            prob_per_class.append(prob)

            if prob > best_prob:
                pred_class = class_name
                best_prob = prob
                
        return pred_class