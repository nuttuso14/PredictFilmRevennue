#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday MAY 30, 2019
@author: Nuntanut Bhooanusas, PhD student in Computer and Communication Engineering

Modified from : Madhuri Suthar, PhD Candidate in Electrical and Computer Engineering, UCLA
"""

# Imports
import numpy as np 
import pandas as pd 
      

# Define useful functions    

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y,tt):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],11) # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(11,1)
        self.y = y
        self.tt =tt
        self.output = np.zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2
        
    def backprop(self):
        #print("backprop")
        #print(type(self.y))
        #print(type(self.output))
        #print("backprop")
        #print(self.y)
        #print(self.output)
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        
    def think(self):
        self.test_layer1 = sigmoid(np.dot(self.tt, self.weights1))
        self.test_layer2 = sigmoid(np.dot(self.test_layer1, self.weights2))
        return self.test_layer2
 
    
idf = pd.read_csv("trainning_neural.csv") 
odf = pd.read_csv("test_neural.csv") 

print("Load input files")
    
trdf = idf.filter(['genre_fq','sequel_movie','director_follower_count_on_twitter','actor_follower_count_on_twitter','actress_follower_count_on_twitter','official_trailer_view_count_on_youtube','official_trailer_comment_count_on_youtube','official_trailer_like_count_on_youtube','official_trailer_dislike_count_on_youtube','movie_rating_on_imdb','sentiment_analysis','movie_rating_on_rotten_tomatoes'],axis=1)
ttdf = odf.filter(['genre_fq','sequel_movie','director_follower_count_on_twitter','actor_follower_count_on_twitter','actress_follower_count_on_twitter','official_trailer_view_count_on_youtube','official_trailer_comment_count_on_youtube','official_trailer_like_count_on_youtube','official_trailer_dislike_count_on_youtube','movie_rating_on_imdb','sentiment_analysis','Predicted_result'],axis=1)

income_sum = idf.gross_income.sum();

idf['gross_income'] = idf['gross_income']/idf.gross_income.sum() 

#print(idf['gross_income'])

X = trdf.to_numpy()
y = idf.gross_income.to_numpy()

y.shape = (len(y), 1)


test = ttdf.to_numpy()
NN = NeuralNetwork(X,y,test)


Lerning_round = 1050000
print("wait a few minutes")
print("There are " ,Lerning_round, " rounds")

for i in range(Lerning_round):
    NN.train(X, y)
  
print("Finish Training")
print ("Input : \n" + str(X))
print ("Actual Output: \n" + str(y))
print ("Predicted Output: \n" + str(NN.feedforward()))
print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
print ("\n")

predict_gross = NN.think()
print(predict_gross)

predicted_income = []
for x in range(0,len(predict_gross)):
    real_income = predict_gross[x]*income_sum
    predicted_income.append(real_income)
    
print(predicted_income)
print(type(predicted_income))

rdf = odf.filter(['title','Predicted_result'])
rdf['rotten_rate'] = odf['rotten_translate']
rdf['predicted_income']=np.asarray(predicted_income)
rdf.to_csv('finalresult.csv',index=False, sep=',')