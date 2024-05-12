# 1155202866
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import chisquare

# Problem 2
def problem_2(n):
    output = ""
    for i in range(1, n*4-2):
        if i % 2 == 1:
            a = max((n*4-2-i), i)
            b = (n*4-3) - a
            for j in range(1, n*4-2):
                if j % 2 == 1:
                    output += "@"
                else:
                    if j <= b+1 or n*4-2-j <= b+1:
                        output += " "
                    else:
                        output += "@"
            output += "\n"
        else:
            for j in range(1, n*4-2):
                if j % 2 == 0:
                    output += " "
                else:
                    c = min(i, n*4-2-i)
                    if j < c or n*4-2-j < c:
                        output += "@"
                    else:
                        output += " "
            output += "\n"
    
    return output

# Problem 3
def problem_3(mat):
    mat_squ = np.square(mat)
    rowsum = np.sum(mat_squ, axis=1)
    colsum = np.sum(mat_squ, axis=0)

    return rowsum, colsum

# Problem 4
def problem_4(survey):
    valid = range(1, 7)
    mask = np.isin(survey, valid)
    valid_nums = np.where(mask, survey, np.nan)
    averages = np.nanmean(valid_nums, axis=0)
    output = np.where(mask, survey, averages)
    
    return output

# Problem 5
def problem_5(dates):
    header = ["date","year","month","day","hour"]
    df = pd.DataFrame(columns=header)
    for date in dates:
        df = df.append({"date": date, "year": date.year , "month": date.month, "day": date.day, "hour": date.hour}, ignore_index=True)
    df['year'] = df['year'].astype("int64")
    df['month'] = df['month'].astype("int64")
    df['day'] = df['day'].astype("int64")
    df['hour'] = df['hour'].astype("int64")
    
    return df

# Problem 6
def problem_6(list_of_sum):
    obs = []
    for i in range(11):
        obs.append(list_of_sum.count(i+2))
    f_obs = np.array(obs)
    f_exp = np.array([1,2,3,4,5,6,5,4,3,2,1])/36*len(list_of_sum)
    chi2, p = chisquare(f_obs, f_exp)

    return p, chi2


if __name__ == "__main__":
    # Testing: Problem 2
    print(problem_2(3))
    print(problem_2(4))

    # Testing: Problem 3
    m = np.array([[1,0,1],[-1,2,-3]])
    r, c = problem_3(m)
    print("row sum :", r)
    print("col sum :", c)
    
    # Testing: Problem 4
    m = np.array([[1,2,3,4,5,6],[3,3,3,3,4,4],[0,4,6,2,7,3],[6,6,6,6,7,6],[2,-1,-1,3,5,4],[5,4,5,4,54,0],[6,6,6,6,6,6]])
    m_processed = problem_4(m)
    print(m_processed)

    # Testing: Problem 5
    start_date = datetime(2023, 9, 1)
    end_date = datetime(2023, 9, 10)
    dates = []
    while start_date <= end_date:
        dates.append(start_date)
        start_date += timedelta(days=1)
    df = problem_5(dates)
    print(df)
    print(df.dtypes)
    
    # Testing: Problem 6
    sum = [2,7,10,10,11,8,5,6,6,5,5,6,6,9,3,7,9,7,10,8]
    p, chi2 = problem_6(sum)
    print("p-value :", p)
    print("chi-square :", chi2)
