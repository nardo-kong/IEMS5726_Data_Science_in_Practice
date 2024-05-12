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
    all_data = []

    # Read and append all data to the list
    for csv_file in input_csv_list:
        data = pd.read_csv(csv_file, header=None, names=['Test1', 'Test2'])
        data['Class'] = csv_file
        all_data.append(data)

    all_data_df = pd.concat(all_data)  # Concatenate all data into a single DataFrame

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))  # Create a subplot for each test

    # Plot the data
    sns.boxplot(x='Class', y='Test1', data=all_data_df, ax=axs[0])
    sns.boxplot(x='Class', y='Test2', data=all_data_df, ax=axs[1])

    axs[0].set_title('Test1')
    axs[1].set_title('Test2')
    for j in range(2):
        axs[j].set_ylabel('Score')
        axs[j].set_ylim(0, 100)
        axs[j].grid(True)

    fig.suptitle('Test result')
    plt.tight_layout()  
    plt.savefig(output, bbox_inches='tight')  # do not show the plt
    plt.close()

# Problem 3
def problem_3(filenames, output="q3.jpg"):
    # write your logic here
    cols = 2
    rows = (len(filenames) + 1)//cols
    fig, axs = plt.subplots(rows, cols, figsize=(10, rows * 5))
    for i in range(rows * cols):
        ax = axs[i//2, i%2]
        if i < len(filenames):
            with open(filenames[i], 'r', encoding='utf-8') as f:
                text = f.read()
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\d', '', text)
                text = text.lower()
                word_cloud = WordCloud(collocations = False, background_color = 'white', random_state=5726).generate(text)
                ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output) # do not show the plt
    plt.close()

# Problem 4
def problem_4(df, output="q4.png"):
    # write your logic here
    df['Boysbackup'] = df['Boys']
    df['Boys'] = df['Boys'] / ( df['Boys'] + df['Girls'] )
    df['Girls'] = df['Girls'] / ( df['Boysbackup'] + df['Girls'] )
    df.drop(columns=['Boysbackup'], inplace=True)

    ax = df.plot(kind='barh', stacked=True, figsize=(10, 5))

    # Label the bars with the percentage values
    for p in ax.patches:
        width = p.get_width()
        plt.text(p.get_x() + width,
                 p.get_y() + p.get_height()/2,
                 f'{width:.4f}',
                 ha='left',
                 va='center')

    plt.title('Passing Percentage')
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.ylabel('Years')

    plt.savefig(output) # do not show the plt
    plt.close()


# Problem 5
import matplotlib.colors as mcolors
def problem_5(n=100, r=1, output="q5.jpg"):
    # write your logic here
    random.seed(4320) # for reproducibility

    square_x = []
    square_y = []
    circle_x = []
    circle_y = []
    
    for _ in range(n):
        side = random.randint(1, 4)
        position = random.uniform(-r, r)
        if side == 1:
            square_x.append(position)
            square_y.append(r)
        elif side == 2:
            square_x.append(r)
            square_y.append(position)
        elif side == 3:
            square_x.append(position)
            square_y.append(-r)
        else:
            square_x.append(-r)
            square_y.append(position)
    
    for _ in range(n):
        angle = random.uniform(0, 2 * np.pi)
        circle_x.append(r * np.cos(angle))
        circle_y.append(r * np.sin(angle))
    
    plt.figure()
    plt.scatter(square_x, square_y, c='#4B0082')

    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#008B8B","#FFEE00"])
    colors = np.linspace(0, 1, len(circle_x))
    plt.scatter(circle_x, circle_y, c=colors, cmap=cmap, label='Circle Cluster')
    
    plt.savefig(output)
    # do not show the plt
    plt.close()

# Problem 6
def problem_6(data, b=10, output="q6.jpg"):
    # write your logic here, model is the trained ridge model
    random.seed(4320) # for reproducibility

    plt.hist(data, bins=b)

    if np.mean(data) < np.median(data):
        plt.gca().yaxis.tick_right()
    else:
        plt.gca().yaxis.tick_left()

    plt.savefig(output)  
    # do not show the plt
    plt.close()


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
    