from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import data_preprocess
import data_train



def make_plot(models,x_test,y_test):
    score = []
    model_name = ["XGBoost", "Logistic_Regression", "SVM_Classifier", "Desicion_Tree"]
    real = 4 * [100]
    for model in models:
        score.append(models[model].score(x_test, y_test) * 100)

    draw = pd.DataFrame()
    draw = draw.reindex(model_name)
    draw["Accuraty"] = score
    draw["real"] = real
    sns.heatmap(draw, annot=True, fmt=".2f", cmap="binary")
    plt.show()


    for model in models:
        y_predict = models[model].predict(x_test)
        v_data = pd.DataFrame({"predict": list(y_predict), "real": list(y_test)})

        sns.catplot(x="predict", y="real", data=v_data)

        plt.show()
        


def show_tree(models,x_test):
    import graphviz
    tree.plot_tree(models["Desicion_Tree"])
    plt.show()




    dot_data = tree.export_graphviz(models["Desicion_Tree"], out_file=None,
                                    feature_names=x_test.columns,
                                    class_names="status",
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)

    graph.render("Tree", view=True, format="png", cleanup=True)




if __name__ == '__main__':
    df = data_preprocess.preprocess()
    x_train, x_test, y_train, y_test = data_preprocess.prepare_for_models(df)
    models = data_train.trainer(x_train, y_train)
    data_train.calculate_accuraty(models, x_test, y_test)
    data_train.make_prediction(models, x_test, y_test)
    make_plot(models,x_test,y_test)
    # you should install the graphwiz
    show_tree(models,x_test)