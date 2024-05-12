# 1155202866
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import random

# Problem 2
class MyDataset(Dataset):
    def __init__(self, X, y):      
        self.X=torch.tensor(X.values, dtype=torch.float32)
        self.y=torch.tensor(y.values, dtype=torch.float32)
 
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def problem_2(df, Xlabel, ylabel, hidden=[3,3,3], test_size=0.3, batch_size=100, learning_rate=0.01, max_epochs=50000):
    # write your logic here, model is the trained ANN model
    model = []
    precision = 0
    recall = 0
    f1score = 0
    random_state = 4320  # default is 4320
    module = []
    
    # YOUR CODE HERE: create the list of modules
    n_features = len(Xlabel)
    module.append(nn.Linear(n_features, hidden[0]))
    module.append(nn.ReLU())
    for i in range(len(hidden)-1):
        module.append(nn.Linear(hidden[i], hidden[i+1]))
        module.append(nn.ReLU())
    
    # the output layer is fixed to 1 neural and sigmoid
    module.append(nn.Linear(hidden[-1],1))
    module.append(nn.Sigmoid())
    # create the nn model based on the list
    model = nn.Sequential(*module)
    
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df[Xlabel], df[ylabel], test_size=test_size, random_state=random_state)
    
    # create the DataLoader for the training set
    mytrain = MyDataset(X_train, y_train)
    train_loader = DataLoader(mytrain, batch_size=batch_size, shuffle=False)

    # use MSE loss function, and SGD optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # training loop
    for epoch in range(max_epochs):
        # you can uncomment the line below, to visualize the slow training process
        # print("Debug: at epoch: ", epoch)
        for data, labels in train_loader:
            # YOUR CODE HERE: training loop
            a = 0 # to prevent empty loop
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

    
    
    # YOUR CODE HERE: follow the training set to create dataloader for testing
    mytest = MyDataset(X_test, y_test)
    test_loader = DataLoader(mytest, batch_size=batch_size, shuffle=False)
    
    # then, calculate the metrics
    y_pred = []
    y_true = []
    for data, labels in test_loader:
        output = model(data)
        predicted_classes = (output > 0.5).float()
        y_pred.extend(predicted_classes.view(-1).numpy())
        y_true.extend(labels.view(-1).numpy())
    
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1score = metrics.f1_score(y_true, y_pred)

    return model, precision, recall, f1score

# Problem 3
def problem_3(image_filename):
    # write your logic here, keypoint and descriptor are BRISK object
    keypoint = 0
    descriptor = 0

    img = cv.imread(image_filename, cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(img, None)
    
    return keypoint, descriptor

# Problem 4
def problem_4(descriptor_list, k=5):
    # write your logic here, visual_words are the cluster centroids
    visual_words = 0
    random_state = 5726
    
    descriptor_array = np.vstack(descriptor_list)
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(descriptor_array)
    visual_words = kmeans.cluster_centers_
    
    return visual_words

# Problem 5
# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class. 
def image_class(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

def find_index(each_feature, centers):
    best = 0
    dist = np.linalg.norm(each_feature-centers[0])
    for k in range(len(centers)):
        if np.linalg.norm(each_feature-centers[k]) < dist:
            best = k
            dist = np.linalg.norm(each_feature-centers[k])
    return best

def problem_5(list_of_image, k=5):
    # write your logic here, visual_bag_of_words is a dictionary
    sift_vectors = {}
    descriptor_list = []
    visual_bag_of_words = {}

    # YOUR CODE HERE:
    # reuse problem 3 and collect the list of all descriptors
    for img in list_of_image:
        kp, des = problem_3(img)
        descriptor_list.append(des)
        sift_vectors[img] = des

    # YOUR CODE HERE:
    # reuse problem 4 and obtain the visual words
    visual_words = problem_4(descriptor_list, k)

    # YOUR CODE HERE:
    # based on the visual words and cluster centers
    # construct the visual bow (dict)
    for img in list_of_image:
        visual_bag_of_words[img] = image_class({img: [sift_vectors[img]]}, visual_words)[img]

    return visual_bag_of_words


# Problem 6
from sklearn.model_selection import KFold
def problem_6(df_X, df_y, kfold=10, alpha=1.0):
    # write your logic here, model is the trained ridge model
    model = 0
    rmse = 0
    random_state = 4320  # default is 4320
    random.seed(random_state) # may be useful

    model = Ridge(alpha=alpha)
    kf = KFold(n_splits=kfold, random_state=random_state, shuffle=True)
    
    squared_errors = []
    scaler = StandardScaler()

    for train_index, test_index in kf.split(df_X):
        X_train, X_test = df_X.iloc[train_index], df_X.iloc[test_index]
        y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        squared_errors.append(metrics.mean_squared_error(y_test, y_pred))

    rmse = np.sqrt(np.mean(squared_errors))

    model.fit(df_X, df_y)
    
    
    return model, rmse


if __name__ == "__main__":
    # Testing: Problem 2
    df = pd.read_csv("creditcard_2023.csv")
    Xlabel = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
    ylabel = ["Class"]
    model, p, r, f = problem_2(df, Xlabel, ylabel, hidden=[3,5,5,3], max_epochs=100)
    print("model: ", model)
    print("precision: ", p)
    print("recall: ", r)
    print("f1-score: ", f)

    # Testing: Problem 3
    kp, des = problem_3("sample1.jpg")
    print(kp)
    print(len(kp))
    print(des)
    print(des.shape)
   
    # Testing: Problem 4
    kp, des = problem_3("sample1.jpg")
    visual_words = problem_4(des)
    print(visual_words)
    print(visual_words.shape)
    
    # Testing: Problem 5
    list_of_image = ["sample1.jpg", "sample2.jpg", "sample3.jpg", "sample4.jpg", "sample5.jpg"]
    vbow = problem_5(list_of_image, 10)
    print(vbow)  
    
    # Testing: Problem 6
    df = pd.read_csv('USA_Housing.csv')
    df_X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
    df_X = pd.DataFrame(StandardScaler().fit_transform(df_X.values))
    df_y = df['Price']
    model, rmse = problem_6(df_X, df_y, kfold=10)
    print("model: ", model)
    print("cv rmse: ", rmse)
