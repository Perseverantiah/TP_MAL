import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def boxplot_(data,col):
    """
    data : pd.DataFrame
    col : str (name of col)
    return : box plot

    """
    
    sns.boxplot(data[col])
    plt.title('{}'.format(col))
    return plt.show()


def boxplot_biv(data,col,target_name='Y1'):
    """
    data : pd.DataFrame
    col : str (name of col)
    target_name (str)
    return : box plot

    """
    sns.boxplot(x=target_name, y=col,data=data)
    plt.xlabel(target_name)
    plt.ylabel('{}'.format(col))
    return plt.show()


def plot_histogram_with_density(data,col, bins=30):
    """
    Function for displaying a histogram using the density function.

    :param data: data table (list or array)
    :param bins: number of bins for histogram (default: 30)
    :param title: chart title (default: “Histogram with density function”)
    """
    plt.figure(figsize=(10, 6))
    
    # Tracer l'histogramme
    sns.histplot(data[col], bins=bins, kde=True, stat="density", color='skyblue', edgecolor='black')
    
    # Configuration des axes et du titre
   # plt.title(title)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def corr_plot(data, title="Corr Matrix", cmap="coolwarm", annot=True):
    
    """
    Function for displaying the correlation matrix of variables in a DataFrame.

    :param data: DataFrame containing data
    :param title: heat map title (default: “Correlation Matrix”)
    :param cmap: color palette (default: “coolwarm”)
    :param annot: displays correlation coefficients in each cell (default: True)
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, square=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.show()
