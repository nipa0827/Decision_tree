#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 23:15:12 2019

@author: suravi
"""

import pandas as pd

'''
training_data = [
        ['Green', 3, 'Mango'],
        ['Yellow', 3, 'Mango'],
        ['Red', 1,'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon']
        ]
'''

training_data = pd.read_csv("weather.csv") 

training_data = training_data.values

#column labels
#these are used only to print the tree
header = ["outlook", "temperature", "humidity", "windy", "label"]

def unique_vals(rows, col):
    #find a unique values for a column in a dataset
    return set([row[col] for row in rows])


#print(unique_vals(training_data,2))
    
def class_counts(rows):
    #counts the number of each type of example in a dataset
    
    counts = {} # a dictionary of label -> count.
    
    for row in rows:
        # in the dataset format, the label is always the last column
        label = row[-1]
        
        if label not in counts:
            counts[label] = 0
            
        counts[label] += 1
        
    return counts

#print(class_counts(training_data))

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

#print(is_numeric("Red"))
    
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    
        
    def match(self, example):
        #compare the feature value in an example to the feature value in the question
        val = example[self.column]
        
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
        
    def __repr__(self):
        #this is just a helper method to print
        # the question in a reduccable format
        
        condition = "=="
        
        if is_numeric(self.value):
            condition = ">="
            
        return "Is %s %s %s" % (header[self.column], condition, str(self.value))
    
def partition(rows, question):
    true_rows, false_rows = [], []
        
    for row in rows:
        if question.match(row):
            true_rows.append(row)
                
        else:
            false_rows.append(row)
                
    return true_rows, false_rows
    
true_rows, false_rows =  partition(training_data, Question(0, 'Red'))
    
def gini(rows):
    
    counts = class_counts(rows)
    impurity = 1
    
    for lbl in counts:
        prob_of_lbl = counts[lbl]/float(len(rows))
        impurity -= prob_of_lbl**2
    
    return impurity

def info_gain(left, right, current_uncertainity):
    
    p = float(len(left))/ (len(left)+len(right))
    return current_uncertainity - p*gini(left) - (1-p) * gini(right)
    

def find_best_split(rows):
    best_gain = 0 #keep track of the best information gain
    best_question = None #keep train of the feature / value that prodced it
    
    current_uncertainity = gini(rows)
    n_features = len(rows[0]) -1
    
    for col in range(n_features):
        values = set([row[col] for row in rows])
        
        for val in values:
            question = Question(col, val)
            
            #try splitting the dataset
            true_rows, false_rows = partition(rows, question)
            
            if len(true_rows)==0 or len(false_rows)==0:
                continue
            
            gain = info_gain(true_rows, false_rows, current_uncertainity)
            
            if gain>= best_gain:
                best_gain, best_question = gain, question
                
    return best_gain, best_question

class Leaf:
    
    def __init__(self, rows):
        self.predictions = class_counts(rows)
        

class Decision_Node:
    
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        
def build_tree(rows):
    #try partitioning the dataset on each of the unique attribute
    #calculate information gain
    #and return the question that produces the highest gain
    gain, question = find_best_split(rows)
        
    #base_case: no further information gain
    #since we can ask no further questions
    #we'll return a leaf
    if gain == 0:
        return Leaf(rows)
        
    #if we reach here, we have found a useful feature / value
    #to partition on.
    true_rows, false_rows = partition(rows, question)
        
    #recursively build the true branch
    true_branch = build_tree(true_rows)
        
    #recursively build the false branch
    false_branch = build_tree(false_rows)
        
        
    return Decision_Node(question, true_branch, false_branch)
        
        
def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing+"Predict", node.predictions)
        return
    
    #print the question at this node
    print(spacing+str(node.question))
    
    #call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing+" ")
    
    #call this function recursively on the false branch
    print(spacing+'--> False: ')
    print_tree(node.false_branch, spacing+" ")
        
        
def classify(row, node):
    
    if isinstance(node, Leaf):
        return node.predictions
    
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl]/total*100))
    
    return probs
        
        
if __name__=='__main__':
    my_tree = build_tree(training_data)
      
    print_tree(my_tree)
    
    #evaluate
    testing_data = [
            ['Green', 3, 'Mango'],
            ['Yellow', 4, 'Mango'],
            ]
    
    for row in testing_data:
        print("Actual: %s, Predicted: %s"% 
              (row[-1], print_leaf(classify(row,my_tree))))
