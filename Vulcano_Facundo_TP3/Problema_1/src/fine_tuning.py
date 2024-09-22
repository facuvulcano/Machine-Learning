import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models import LogisticRegression, LogisticRegressionUndersampling, LogisticRegressionOversampling, LogisticRegressionCostReWeighting, LogisticRegressionSmote
from data_splitting import cross_val
from binary_metrics import ClassificationMetrics
from tqdm import tqdm
from tabulate import tabulate

def find_best_lambda(df, regresion_class, threshold, rebalancing_method=None):

    MAX_ITER = 10
    lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    global_results = {lambda_ : 0 for lambda_ in lambdas}
    coef_norm_avg = {lambda_ : 0 for lambda_ in lambdas}
    n_folds = 5
    original_df = df

    for iter in tqdm(range(MAX_ITER), desc="Outer Iteration", unit="iteration"):
        for lambda_ in lambdas:
            df = original_df.copy()
            
            reg_model = regresion_class(
                df=df,
                threshold=threshold,
                max_iter=1000,
                learning_rate=0.1,
                lambda_penalty=lambda_
            )
            if rebalancing_method is not None:
                balanced_df = rebalancing_method(reg_model)
            else:
                balanced_df = df
            
            #data splitting
            training, validation = cross_val(balanced_df, 'target', n_folds)

            fold_f_scores = []
            fold_coef_norms = []
            y_val_cv_lru = []
            y_val_cv_pred_lru = []

            for (train, val) in zip(training, validation):
                X_train_cv, y_train_cv = train[0], train[1].to_numpy()
                X_val_cv, y_val_cv = val[0], val[1].to_numpy()

                #model training
                reg_model.fit(X_train_cv, y_train_cv)

                #predicitons on validation data
                y_pred_val_cv = reg_model.predict(X_val_cv)
                predicted_probabilites_val_cv = reg_model.predict_proba(X_val_cv)

                y_val_cv_lru.append(y_val_cv)
                y_val_cv_pred_lru.append(y_pred_val_cv)

                #calculation of f-score metric
                metrics = ClassificationMetrics(y_val_cv, y_pred_val_cv, predicted_probabilites_val_cv)
                f_score = metrics.f_score()
                fold_f_scores.append(f_score)

                coef_norm = np.linalg.norm(reg_model.coef_)
                fold_coef_norms.append(coef_norm)

            avg_f2_score = np.mean(fold_f_scores)
            avg_coef_norm = np.mean(fold_coef_norms)

            global_results[lambda_] += avg_f2_score
            coef_norm_avg[lambda_] += avg_coef_norm


    for lambda_ in global_results:
        global_results[lambda_] /= MAX_ITER
        coef_norm_avg[lambda_] /= MAX_ITER

    title = 'RESULTADOS DE F-SCORE PARA DISTINTOS LAMBDAS'
    print(f'{title}')

    data = {
        'Lambda' : [key for key, _ in global_results.items()],
        'F-Score' : [value for _, value in global_results.items()]
    }
    data_df = pd.DataFrame(data)
    print(tabulate(data_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    best_lambda = max(global_results, key=global_results.get)
    print(f'El mejor lambda es {best_lambda} con un f-score promedio de {global_results[best_lambda]:.4f}')

    plt.plot(list(coef_norm_avg.keys()), list(coef_norm_avg.values()), marker = 'o')
    plt.xscale('log')
    plt.xlabel('Lambda (Regularization Strength)')
    plt.ylabel('Coefficient Norm')
    plt.title('Effect of regularization on coefficient magnitudes')
    plt.show()

    return best_lambda