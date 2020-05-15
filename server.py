#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import math

app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    # TODO: Start sampling (12 points)
    sample_points_X, sample_points_Y = [], []

    X = points_X
    Y = points_Y
    dist = np.cumsum(np.sqrt(np.ediff1d(X, to_begin=0) ** 2 + np.ediff1d(Y, to_begin=0) ** 2))
    dist = dist / dist[-1]
    dist_x, dist_y = interp1d(dist, X), interp1d(dist, Y)
    alpha_value = np.linspace(0, 1, 100)
    sample_points_X, sample_points_Y = dist_x(alpha_value), dist_y(alpha_value)
    print("Here is the input: ",sample_points_X)
    print("Let's look at data before sampling: ",X)
    if(type(X[0]) is list):
        sample_points_X[0][0]=X[0][0]
    if(type(Y[0]) is list):
        sample_points_Y[0][0]=Y[0][0]
    return sample_points_X, sample_points_Y



# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 18
    # TODO: Do pruning (12 points)
    # print("gesture_points_X:", gesture_points_X)
    print("Pruning Begins")
    for i in range(0, len(template_sample_points_X)):
        # for j in template_sample_points_Y
            start_x_ges = gesture_points_X[0][0]
            start_y_ges = gesture_points_Y[0][0]
            end_x_ges= gesture_points_X[0][-1]
            end_y_ges= gesture_points_Y[0][-1]

            start_x_temp = template_sample_points_X[i][0]
            start_y_temp = template_sample_points_Y[i][0]
            end_x_temp = template_sample_points_X[i][-1]
            end_y_temp = template_sample_points_Y[i][-1]

            # print("Points considered:", x_start, y_start, comp_x_start, comp_y_start)

            if math.sqrt((start_x_ges - start_x_temp)**2 + (start_y_ges - start_y_temp)**2) <= threshold and math.sqrt((end_x_ges - end_x_temp)**2 + (end_y_ges - end_y_temp)**2) <= threshold:
                valid_template_sample_points_X.append(template_sample_points_X[i])
                valid_template_sample_points_Y.append(template_sample_points_Y[i])
                valid_words.append((words[i], probabilities[words[i]]))

    print ("Valid words found = ", len(valid_words))
    for (word,p) in valid_words:
        print(word, p)

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    L = 1

    # TODO: Calculate shape scores (12 points)
    for i in range(len(valid_template_sample_points_X)):
        sum = 0
        for j in range(100):
            if(math.isnan(math.sqrt(pow((gesture_sample_points_X[0][j] - valid_template_sample_points_X[i][j]),2) + pow((gesture_sample_points_Y[0][j] - valid_template_sample_points_Y[i][j]),2)))):
                print("Invalid: ",gesture_sample_points_X[0][j],valid_template_sample_points_X[i][j],gesture_sample_points_Y[0][j],valid_template_sample_points_Y[i][j])
            else:
                sum = sum + math.sqrt(pow((gesture_sample_points_X[0][j] - valid_template_sample_points_X[i][j]),2) + pow((gesture_sample_points_Y[0][j] - valid_template_sample_points_Y[i][j]),2))

        shape_scores.append(sum/100)

    print(" Shape score of 1 = ",shape_scores[0])
    return shape_scores


def helper(x,y,points_X,points_Y):
    arr = []
    for i in range(len(points_X)):
        arr.append(math.sqrt((points_Y[i]-y)**2 + (points_X[i]-x)**2))
    return min(arr)

def location_score_helper1(gesture_sample_points_X,gesture_sample_points_Y,template_X,template_Y,r):
    sum = 0
    for i in range(len(gesture_sample_points_X)):
        sum = sum + max(helper(gesture_sample_points_X[0][i],gesture_sample_points_Y[0][i],template_X,template_Y)-r,0)
    return sum


def location_score_helper2(x1,y1,x2,y2,D):
    if(D == 0):
        return 0
    else:
        return math.sqrt((y2-y1)**2 + (x2-x1)**2)

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # TODO: Calculate location scores (12 points)
    values = []
    for i in range(100):
        values.append(0.01)
    # Calculating location_scores.
    # print(" Hello : ",valid_template_sample_points_X[i])
    for i in range(len(valid_template_sample_points_X)):
        sum = 0
        D = location_score_helper1(gesture_sample_points_X,gesture_sample_points_Y,valid_template_sample_points_X[i],valid_template_sample_points_Y[i],radius)
        for j in range(len(gesture_sample_points_X)):
            sum = sum + values[j] * location_score_helper2(gesture_sample_points_X[0][j],gesture_sample_points_Y[0][j],valid_template_sample_points_X[i][j],valid_template_sample_points_Y[i][j],D)
        location_scores.append(sum)
    # for i in
    print(" Location score of 1 = ", location_scores[0])
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.85
    # TODO: Set your own location weight
    location_coef = 0.15
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    # TODO: Set your own range.
    n = len(valid_words)
    if n >= 10:
        n=7
    elif n < 10 and n > 5:
        n=7
    else:
        n=len(valid_words)
    # TODO: Get the best word (12 points)
    a = min(integration_scores)
    b = integration_scores.index(a)
    for i in range(0,n):
        print(" Words = ",valid_words[i][0]," ",integration_scores[i])
    return valid_words[b][0]


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    print("Gest X: ", gesture_points_X)
    print("Gest Y: ", gesture_points_Y)
    gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()


# In[ ]:




