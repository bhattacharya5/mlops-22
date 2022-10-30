# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree, model_selection
import pdb
import statistics


from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)
from joblib import dump, load

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

max_depth_list = [2, 10, 20, 50, 100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)

del digits

# define the evaluation metric
metric_list = [metrics.accuracy_score, macro_f1]
h_metric = metrics.accuracy_score

n_cv = 5
results = {}
for n in range(n_cv):

    if n == 0 :
        train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1        
    elif n == 1 :
        train_frac, dev_frac, test_frac = 0.7, 0.2, 0.1      
    elif n == 2 :
        train_frac, dev_frac, test_frac = 0.6, 0.2, 0.2        
    elif n == 3 :
        train_frac, dev_frac, test_frac = 0.5, 0.3, 0.2        
    elif n == 4 :
        train_frac, dev_frac, test_frac = 0.5, 0.2, 0.3
        

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )

    # PART: Define the model
    # Create a classifier: a support vector classifier
    models_of_choice = {
        "svm": svm.SVC(),
        "decision_tree": tree.DecisionTreeClassifier(),
    }

    for clf_name in models_of_choice:
        clf = models_of_choice[clf_name]

        print("[{}] Running hyper param tuning for {}".format(n,clf_name))

        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
        )

        # 2. load the best_model
        best_model = load(actual_model_path)

        # PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = best_model.predict(x_test)
        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})       

       
        
        # 3. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

        #5. Identify if there is more to the classifier comparison than just the numbers? -- Can you somehow compare the predicted labels, instead of the metrics
        # Quantitative Measurement of Performance
        matches = (predicted == y_test)
        print('Matches :', matches.sum())
        print('Expected : ', len(matches))
        print('Quantitative Measurement of Performance : ', matches.sum() / float(len(matches)),"\n")


print("\n\n",results)

#4.Mean and standard deviation of metrics performance
print(
    f"\n Mean of model performance metrics accuracy is {clf_name} is : " , (statistics.mean([x['accuracy_score'] for x in results[clf_name]]))
    )


print(
    f"Stadard Deviation of model performance metrics accuracy is {clf_name} is : " , (statistics.mean([x['accuracy_score'] for x in results[clf_name]])), "\n"
)
