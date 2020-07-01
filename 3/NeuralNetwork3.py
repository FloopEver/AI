# -*- coding: utf-8 -*-
import numpy as np
import csv
import random

# read image, every row is an image
def read_image(file):
    
    with open(file,'r') as csvfile:
        reader = csv.reader(csvfile)
        data_lst = []
        for line in reader:
            data_lst.append([int(x) / 255 for x in line])
    data = np.array(data_lst)

    return data


#read label, every row is a label
def read_label(file):
    
    with open(file,'r') as csvfile:
        reader = csv.reader(csvfile)
        data_lst = []
        for line in reader:
            d = [int(x) for x in line]
            newd = np.zeros(10)
            newd[d[0]] = 1
            data_lst.append(newd)
    data = np.array(data_lst)
    
    return data


class NeuralNetwork:
    def __init__(self, inputs, hidden1, hidden2, outputs, learningrate):
        
        self.i = inputs
        self.h1 = hidden1
        self.h2 = hidden2
        self.o = outputs
        self.lr = learningrate
        # initialize weight matrices
        self.wih = (np.random.normal(0.0, pow(self.h1, -0.5), (self.h1, self.i)))
        self.whh = (np.random.normal(0.0, pow(self.h2, -0.5), (self.h2, self.h1)))
        self.who = (np.random.normal(0.0, pow(self.o, -0.5), (self.o, self.h2)))


    def sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))
    
    
    def softmax(self, x):
        
        xx = np.max(x)
        exp_x = np.exp(x - xx)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        
        return y
    
    
    def test(self, images):
        
        inputs = np.array(images, ndmin=2).T
        # from input to hidden1
        h1_in = np.dot(self.wih, inputs)
        h1_out = self.sigmoid(h1_in)
        # from hidden1 to hidden2
        h2_in = np.dot(self.whh, h1_out)
        h2_out = self.sigmoid(h2_in)
        # from hidden2 to output
        o_in = np.dot(self.who, h2_out)
        o_out = self.softmax(o_in)
        
        return o_out
    
    
    def train(self, images, labels):
        
    # Forward
        inputs = np.array(images, ndmin=2).T
        # from input to hidden1
        h1_in = np.dot(self.wih, inputs)
        h1_out = self.sigmoid(h1_in)
        # from hidden1 to hidden2
        h2_in = np.dot(self.whh, h1_out)
        h2_out = self.sigmoid(h2_in)
        # from hidden2 to output
        o_in = np.dot(self.who, h2_out)
        o_out = self.softmax(o_in)
    
    # Backward
        answer = np.array(labels, ndmin=2).T
        # errors
        o_errors = answer - o_out
        h2_errors = np.dot(self.who.T, o_errors * o_out * (1.0 - o_out))
        h1_errors = np.dot(self.whh.T, h2_errors * h1_out * (1.0 - h1_out))
        # update weights
        self.who += self.lr * np.dot((o_errors * o_out * (1.0 - o_out)), np.transpose(h2_out))
        self.whh += self.lr * np.dot((h2_errors * h2_out * (1.0 - h2_out)), np.transpose(h1_out))
        self.wih += self.lr * np.dot((h1_errors * h1_out * (1.0 - h1_out)), np.transpose(inputs))
       
        return
        

def main():
    
    epochs = 7
    my_answer = []
    count = 0
    
    train_image = read_image('train_image.csv')
    train_label = read_label('train_label.csv')
    test_image = read_image('test_image.csv')
    test_label = read_label('test_label.csv')
    
    num = train_image.shape[0]
    num = 10000

    
    n = NeuralNetwork(784, 100, 100, 10, 0.18)
    
    for train_times in range(epochs):
        ran = random.sample(range(0, num), num)
        for i in range(num):
            n.train(train_image[ran[i]], train_label[ran[i]])
    
    for i in range(test_image.shape[0]):
        a = n.test(test_image[i])
        a_answer = np.argmax(a)
        my_answer.append(a_answer)

        ta = test_label[i]
        aa = np.argmax(a)
        taa = np.argmax(ta)
        if aa == taa:
            count += 1
    
    print(count)

    with open('test_predictions.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in my_answer:
            writer.writerow([row])
        
        

if __name__ == '__main__':
    main()
