from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as py

decision_tree_classifier = DecisionTreeClassifier()

training_dataset = pd.read_csv("Datasets/Arabic_Dataset/TrainImages_60kx784.csv").values
training_labels = pd.read_csv("Datasets/Arabic_Dataset/TrainLabel_60kx1.csv").values

test_dataset = pd.read_csv("Datasets/Arabic_Dataset/TestImages_10kx784.csv").values

training_data = training_dataset[0:60000, 0:784]
training_label = training_labels[0:]

decision_tree_classifier.fit(training_data, training_label)

# print(decision_tree_classifier.predict([test_dataset[8]]))



