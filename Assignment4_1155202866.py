# <Your student ID>
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import re
from wordcloud import WordCloud

# Problem 2
def problem_2(input_csv_list, output="q2.jpg"):
    # write your logic here

    plt.savefig(output) # do not show the plt

# Problem 3
def problem_3(filenames, output="q3.jpg"):
    # write your logic here

    plt.savefig(output) # do not show the plt

# Problem 4
def problem_4(df, output="q4.png"):
    # write your logic here

    plt.savefig(output) # do not show the plt

# Problem 5
def problem_5(n=100, r=1, output="q5.jpg"):
    # write your logic here
    random.seed(4320) # for reproducibility

    plt.savefig(output)
    # do not show the plt

# Problem 6
def problem_6(data, b=10, output="q6.jpg"):
    # write your logic here, model is the trained ridge model
    random.seed(4320) # for reproducibility

    plt.savefig(output)  
    # do not show the plt


if __name__ == "__main__":
    # Testing: Problem 2
    filenames = ["classA.csv","classB.csv","classC.csv"]
    problem_2(filenames, "q2.png")

    # Testing: Problem 3
    plt.clf()
    filenames = ["paragraph1.txt","paragraph2.txt","paragraph3.txt"]
    problem_3(filenames, "q3.png")
    
    # Testing: Problem 4
    plt.clf()
    students = pd.DataFrame({'Boys': [67, 78], 'Girls': [72, 80], }, index=['First Year', 'Second Year'])
    problem_4(students, "q4.jpg")

    # Testing: Problem 5
    plt.clf()
    problem_5(n=200, r=5, output="cluster.jpg")

    # Testing: Problem 6
    plt.clf()
    data = np.array([1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,4,4,5,5,6,7,8,9,10,10])
    problem_6(data, b=10, output="hist_left.jpg")
    plt.clf()
    data = 11 - data
    problem_6(data, b=5, output="hist_right.jpg")
    