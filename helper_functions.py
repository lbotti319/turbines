import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_heatmap(df):

    fig, ax = plt.subplots(figsize=(10,10))

    im = ax.imshow(np.abs(df.corr()))
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)

    for i, c_i in enumerate(df.columns):
        for j, c_j in enumerate(df.columns):
            text = ax.text(
                j, i,
                np.abs(round(df.corr().loc[c_i, c_j],2)),
                ha="center", va="center", color='r')

    plt.show()

    
def linear_models(df): 
    results = {}
    for column in df.columns:
        linear = LinearRegression(fit_intercept=False)
        y = df[column]
        X = df.drop(column, axis=1)
        linear.fit(X,y)
        result = {
            "model": linear,
            "score": linear.score(X,y),
            "coefficients": pd.Series(linear.coef_, index=X.columns)
        }
        results[column] = result
    return results


def plot_3D(df, cols, ax, clusters=None):
    if clusters is None:
        ax.scatter3D(
            df[cols[0]],
            df[cols[1]],
            df[cols[2]]
        )
    else:
        ax.scatter3D(
            df[cols[0]],
            df[cols[1]],
            df[cols[2]],
            c=clusters,
            cmap='Set2'
        )
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

def plot_2D(df, cols, ax, clusters=None):
    if clusters is None:
        ax.scatter(
            df[cols[0]],
            df[cols[1]],
        )
    else:
        ax.scatter(
            df[cols[0]],
            df[cols[1]],
            c=clusters,
            cmap='Set2'
        )
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
