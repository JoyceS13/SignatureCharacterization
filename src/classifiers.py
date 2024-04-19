import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import joblib
import shap
import os

# Load the data
def load_data(input_file):
    df = pd.read_csv(input_file, delimiter="\t", index_col=0)
    X = df.drop(columns=['Data Label']).iloc[1:, :]
    y = df['Data Label'].iloc[1:]
    return X, y

def one_vs_all(y, label):
    y_new = []
    for i in range(len(y)):
        if y[i] == label:
            y_new.append(1)
        else:
            y_new.append(0)
    return y_new
        
def random_forest(X_train, X_test, y_train, y_test, scoring='accuracy'):
    rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #tune hyperparameters with grid search and cross validation
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }
    CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scoring, cv= 5)
    # model is made with the best hyperparameters
    CV_rf.fit(X_train, y_train)
    rf = CV_rf.best_estimator_
    y_pred = rf.predict(X_test)
    return rf, y_test, y_pred

def neural_network(X_train, X_test, y_train, y_test, scoring='accuracy'):
    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    #tune hyperparameters with grid search and cross validation
    param_grid = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100, 100, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    CV_nn = GridSearchCV(estimator=nn, param_grid=param_grid, scoring=scoring, cv= 5)
    # model is made with the best hyperparameters
    CV_nn.fit(X_train, y_train)
    nn = CV_nn.best_estimator_
    y_pred = nn.predict(X_test)
    return nn, y_test, y_pred

def calculate_performance_metrics(y_test, y_pred):
    # Calculate performance metrics for multi-class classification
    accuracy = accuracy_score(y_test, y_pred)
    micro_precision = sk.metrics.precision_score(y_test, y_pred, average='micro')
    micro_recall = sk.metrics.recall_score(y_test, y_pred, average='micro')
    micro_f1 = sk.metrics.f1_score(y_test, y_pred, average='micro')
    macro_precision = sk.metrics.precision_score(y_test, y_pred, average='macro')
    macro_recall = sk.metrics.recall_score(y_test, y_pred, average='macro')
    macro_f1 = sk.metrics.f1_score(y_test, y_pred, average='macro')
    if len(set(y_test)) == 2:
        # Calculate performance metrics for binary classification
        binary_precision = sk.metrics.precision_score(y_test, y_pred)
        binary_recall = sk.metrics.recall_score(y_test, y_pred)
        binary_f1 = sk.metrics.f1_score(y_test, y_pred)
        return accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, binary_precision, binary_recall, binary_f1
    return accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1

def feature_importance(model, X, y, output_folder, data_name="", scoring='accuracy'):
    model_name = model.__class__.__name__
    result = permutation_importance(model, X, y, scoring= scoring , n_repeats=10, random_state=0, n_jobs=-1)
    # plot importance in bar chart
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()
    #save plot to file
    plt.savefig(output_folder + '/' + model_name + data_name + '_feature_importance.png')
    #save importance values to file
    with open(output_folder + '/' + model_name + data_name + '_feature_importance.txt', 'w') as f:
        for i in sorted_idx:
            f.write('{}: {}\n'.format(X.columns[i], result.importances_mean[i]))
            
def calculate_shap_rf(rf, X):
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    return shap_values

def calculate_shap_nn(nn, X):
    X_summary = shap.kmeans(X, 50)
    explainer = shap.KernelExplainer(nn.predict, X_summary)
    shap_values = explainer.shap_values(X_summary)
    return shap_values

def plot_shap(shap_values, X, output_folder, model_name):
    class_names = ["BRCA", "LUNG", "STAD", "SKCM"]
    feature_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    shap.summary_plot(shap_values, X.values, plot_type="bar", class_names= class_names, feature_names = X.columns)
    plt.savefig(output_folder + '/' + model_name + '_shap_summary_plot.png')
    shap.summary_plot(shap_values, X)
    plt.savefig(output_folder + '/' + model_name +  '_shap_summary_plot_values.png')

def main(input_file, output_folder):
    #if output folder does not exist, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    X, y = load_data(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf, y_test_rf, y_pred_rf = random_forest(X_train, X_test, y_train, y_test, scoring='f1_macro')
    nn, y_test_nn, y_pred_nn = neural_network(X_train, X_test, y_train, y_test, scoring='f1_macro')
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    accuracy_rf, micro_precision_rf, micro_recall_rf, micro_f1_rf, macro_precision_rf, macro_recall_rf, macro_f1_rf = calculate_performance_metrics(y_test_rf, y_pred_rf)
    accuracy_nn, micro_precision_nn, micro_recall_nn, micro_f1_nn, macro_precision_nn, macro_recall_nn, macro_f1_nn = calculate_performance_metrics(y_test_nn, y_pred_nn)
    print('Random Forest:')
    print('Accuracy: {}'.format(accuracy_rf))
    print('Micro Precision: {}'.format(micro_precision_rf))
    print('Micro Recall: {}'.format(micro_recall_rf))
    print('Micro F1: {}'.format(micro_f1_rf))
    print('Macro Precision: {}'.format(macro_precision_rf))
    print('Macro Recall: {}'.format(macro_recall_rf))
    print('Macro F1: {}'.format(macro_f1_rf))
    print('Neural Network:')
    print('Accuracy: {}'.format(accuracy_nn))
    print('Micro Precision: {}'.format(micro_precision_nn))
    print('Micro Recall: {}'.format(micro_recall_nn))
    print('Micro F1: {}'.format(micro_f1_nn))
    print('Macro Precision: {}'.format(macro_precision_nn))
    print('Macro Recall: {}'.format(macro_recall_nn))
    print('Macro F1: {}'.format(macro_f1_nn))
    #save metrics to file
    with open(output_folder + '/metrics.txt', 'w') as f:
        f.write('Random Forest:\n')
        f.write('Accuracy: {}\n'.format(accuracy_rf))
        f.write('Micro Precision: {}\n'.format(micro_precision_rf))
        f.write('Micro Recall: {}\n'.format(micro_recall_rf))
        f.write('Micro F1: {}\n'.format(micro_f1_rf))
        f.write('Macro Precision: {}\n'.format(macro_precision_rf))
        f.write('Macro Recall: {}\n'.format(macro_recall_rf))
        f.write('Macro F1: {}\n'.format(macro_f1_rf))
        f.write('Neural Network:\n')
        f.write('Accuracy: {}\n'.format(accuracy_nn))
        f.write('Micro Precision: {}\n'.format(micro_precision_nn))
        f.write('Micro Recall: {}\n'.format(micro_recall_nn))
        f.write('Micro F1: {}\n'.format(micro_f1_nn))
        f.write('Macro Precision: {}\n'.format(macro_precision_nn))
        f.write('Macro Recall: {}\n'.format(macro_recall_nn))
        f.write('Macro F1: {}\n'.format(macro_f1_nn))
    #save models to file
    joblib.dump(rf, output_folder + '/random_forest.pkl')
    joblib.dump(nn, output_folder + '/neural_network.pkl')
    #save hyperparameters to file
    with open(output_folder + '/hyperparameters.txt', 'w') as f:
        f.write('Random Forest:\n')
        f.write(str(rf.get_params()))
        f.write('\nNeural Network:\n')
        f.write(str(nn.get_params()))
    #calculate feature importance
    feature_importance(rf, X, y, output_folder, scoring='f1_macro')
    feature_importance(nn, X, y, output_folder, scoring='f1_macro')
    # shap_values_rf = calculate_shap_rf(rf, X)
    # shap_values_nn = calculate_shap_nn(nn, X)
    # plot_shap(shap_values_rf, X, output_folder, model_name='random_forest')
    # plot_shap(shap_values_nn, X, output_folder, model_name='neural_network')
    
def main_one_vs_all(input_file, output_folder):
    #if output folder does not exist, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data_sets = ['SKCM', 'BRCA', 'LUNG', 'STAD']
    X,y = load_data(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for data_set in data_sets:
        y_new_train = one_vs_all(y_train, data_set)
        y_new_test = one_vs_all(y_test, data_set)
        rf, y_test_rf, y_pred_rf = random_forest(X_train, X_test, y_new_train, y_new_test, scoring='f1')
        nn, y_test_nn, y_pred_nn = neural_network(X_train, X_test, y_new_train, y_new_test, scoring='f1')
        X = pd.concat([X_train, X_test])
        y_new = y_new_train + y_new_test
        accuracy_rf, micro_precision_rf, micro_recall_rf, micro_f1_rf, macro_precision_rf, macro_recall_rf, macro_f1_rf, binary_precision, binary_recall, binary_f1 = calculate_performance_metrics(y_test_rf, y_pred_rf)
        accuracy_nn, micro_precision_nn, micro_recall_nn, micro_f1_nn, macro_precision_nn, macro_recall_nn, macro_f1_nn, binary_precision, binary_recall, binary_f1 = calculate_performance_metrics(y_test_nn, y_pred_nn)
        #save metrics to file
        with open(output_folder + '/metrics_' + data_set + '.txt', 'w') as f:
            f.write('Random Forest:\n')
            f.write('Accuracy: {}\n'.format(accuracy_rf))
            f.write('Micro Precision: {}\n'.format(micro_precision_rf))
            f.write('Micro Recall: {}\n'.format(micro_recall_rf))
            f.write('Micro F1: {}\n'.format(micro_f1_rf))
            f.write('Macro Precision: {}\n'.format(macro_precision_rf))
            f.write('Macro Recall: {}\n'.format(macro_recall_rf))
            f.write('Macro F1: {}\n'.format(macro_f1_rf))
            f.write('Binary Precision: {}\n'.format(binary_precision))
            f.write('Binary Recall: {}\n'.format(binary_recall))
            f.write('Binary F1: {}\n'.format(binary_f1))
            f.write('Neural Network:\n')
            f.write('Accuracy: {}\n'.format(accuracy_nn))
            f.write('Micro Precision: {}\n'.format(micro_precision_nn))
            f.write('Micro Recall: {}\n'.format(micro_recall_nn))
            f.write('Micro F1: {}\n'.format(micro_f1_nn))
            f.write('Macro Precision: {}\n'.format(macro_precision_nn))
            f.write('Macro Recall: {}\n'.format(macro_recall_nn))
            f.write('Macro F1: {}\n'.format(macro_f1_nn))
            f.write('Binary Precision: {}\n'.format(binary_precision))
            f.write('Binary Recall: {}\n'.format(binary_recall))
            f.write('Binary F1: {}\n'.format(binary_f1))
        #save models to file
        joblib.dump(rf, output_folder + '/random_forest_' + data_set + '.pkl')
        joblib.dump(nn, output_folder + '/neural_network_' + data_set + '.pkl')
        #save hyperparameters to file
        with open(output_folder + '/hyperparameters_' + data_set + '.txt', 'w') as f:
            f.write('Random Forest:\n')
            f.write(str(rf.get_params()))
            f.write('\nNeural Network:\n')
            f.write(str(nn.get_params()))
        #calculate feature importance
        feature_importance(rf, X, y_new, output_folder, data_name = '_' + data_set, scoring='f1')
        feature_importance(nn, X, y_new, output_folder, data_name = '_' + data_set, scoring='f1')
        
    
def retrieve_models(model_folder):
    rf = joblib.load(model_folder + '/random_forest.pkl')
    nn = joblib.load(model_folder + '/neural_network.pkl')
    return rf, nn
    
if __name__ == '__main__':
    #main('data/labeled_data_11.txt', "results_classifiers_11")
    #main_one_vs_all('data/labeled_data_11.txt', "results_classifiers_11/one_v_all")
    rf, nn = retrieve_models("results_classifiers")
    X, y = load_data('data/labeled_data_test.txt')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    shap_values_rf = calculate_shap_rf(rf, X_train)
    plot_shap(shap_values_rf, X_train, "results_classifiers", model_name='random_forest')
    shap_values_nn = calculate_shap_nn(nn, X_train)
    plot_shap(shap_values_nn, X_train, "results_classifiers", model_name='neural_network')