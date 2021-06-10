import pandas as pd
import numpy as np

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


def is_null(df):
    for model in df.iteritems():
        if model == np.nan:
          return True
    return False

def missing_value(df):
    imp = SimpleImputer(strategy="most_frequent",missing_value="np.nan",fill_value="numeric")
    imp_fit = imp.fit_transform(df)
    tmp = df.columns
    df = pd.DataFrame(imp_fit,columns=tmp)

    print("\nMissing values appended, strategy: most_frequent")

    return df


def preprocess():

    df = pd.read_csv("parkinsons.csv")
    df.info()

    print(df.shape)

    print(df["status"].describe())
    # outlier yok gibi görünüyor
    df.head()

    """# Yeni Bölüm"""

    sns.heatmap(df.corr());
    plt.show()

    # data üzerinde korrelasyon spread1,spread2 ve ppe featureslerinde yoğunlukta headmap üzerinden görebiliriz.

    """# Yeni Bölüm"""

    df_lookup = df.drop(["name"], axis=1)

    for dimension in df_lookup.columns:
        if dimension == "status":
            continue
        sns.boxplot(x="status", y=dimension, data=df_lookup)
        plt.show()

    # spread1,spread2 ve ppe featureları güzel ayrım yapmış. Değerleri yüksek olanlar genelde hasta düşük olanlar ise genelde hasta değil çıkarımını yapabilirz bu noktada.

    for dimension in df_lookup.columns:
        if dimension == "status":
            continue

        sns.lineplot(x="status", y=dimension, color="blue", data=df_lookup)
        plt.show()


    is_null(df)
    #missing value yok

    return df

def scale_data(X):
    tmp = X.columns
    scale = StandardScaler()
    scale.fit(X)
    x = scale.transform(X)
    X = pd.DataFrame(x,columns = tmp)
    return X

def prepare_for_models(df,train_size=0.75,test_size=0.25):
    X = df.drop(["name", "status"], axis=1)
    y = df["status"]

    X = scale_data(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, shuffle=True, stratify=y)

    return x_train, x_test, y_train, y_test

