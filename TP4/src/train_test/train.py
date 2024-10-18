print("El script train.py ha comenzado a ejecutarse")

import sys
import copy
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append('/home/facuvulcano/Machine-Learning/TP4/src')

from models.NN import MLP
from metrics.rmse import rmse
from metrics.mae import mae
from metrics.r2 import r2
from utils.logger import setup_logger
from schedulers.learning_rate_schedulers import LearningRateScheduler
from loss.mse_loss import total_loss
from models.optimizers.sgd import SGD
from models.optimizers.sgd_momentum import SGDMomentum
from models.optimizers.adam import Adam
from models.optimizers.mini_batch_sgd import MiniBatchSGD



print("modulos importados")

def train_model(args, X_train, X_val, y_train, y_val, y_min, y_max):
    print("entrando a funcion")

    logger = setup_logger('train_logger', args.log_file)

    # Definicion del scheduler
    scheduler = LearningRateScheduler(
        initial_lr=1e-7,
        scheduler_type=args.scheduler_type,
        final_lr=args.final_lr,
        power=args.power,
        decay_rate=args.decay_rate
    )

    # Inicializacion del modelo
    input_size = X_train.shape[1]
    model = MLP(input_size, [10, 8, 4, 1])
    num_params = len(model.parameters())
    print(f'Cantidad de parametros: {num_params}')
    logger.info(f'Cantidad de parametros: {num_params}')

    if args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.learning_rate)
    elif  args.optimizer == 'sgd_momentum':
        optimizer = SGDMomentum(learning_rate=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2)
    elif args.optimizer == 'mini_batch_sgd':
        optimizer = MiniBatchSGD(learning_rate=args.learning_rate, batch_size=args.batch_size)
    else:
        raise ValueError(f"Optimizador {args.optimizer} no reconocido")

    # Inicializacion de listas para almacenar perdidas y metricas
    train_losses = []
    val_losses = []
    learning_rates = []
    rmse_vals = []
    mae_vals = []
    r2_vals = []

    # Early Stopping
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    best_model = None

    for epoch in range(1, args.num_epochs + 1):
        print(f'Comenzando la epoca {epoch}')
        current_lr = scheduler.get_lr(epoch, args.num_epochs)
        learning_rates.append(current_lr)

        epoch_loss = 0.0

        if hasattr(optimizer, 'lr'):
            optimizer.lr = current_lr
        elif isinstance(optimizer, Adam):
            optimizer.lr = current_lr
        
        # Entrenamiento
        for i in range(len(X_train)):
            x = X_train.iloc[i].values
            y_true = y_train[i]
            y_pred = model(x)
            #print(f'Prediccion: {y_pred}, Verdadero: {y_true}')

            # Calcular la perdida total
            loss = total_loss(y_true, y_pred, model, args.lambda_reg)
            epoch_loss += loss.data

            # Backpropagation
            loss.backward()

            # Obtener parametros y gradientes
            params = model.parameters()
            grads = [p.grad for p in params]

            optimizer.update(params, grads)
            
            # Reinicializar gradientes
            model.zero_grad()

        # Calcular perdida promedio de entrenamiento
        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)

        # Validacion
        val_loss = 0.0
        y_val_pred_norm = []
        for i in range(len(X_val)):
            x = X_val.iloc[i].values
            y_true = y_val[i]
            y_pred_norm = model(x)
            y_val_pred_norm.append(y_pred_norm.data)
            val_loss += (y_pred_norm.data - y_true) ** 2
        val_loss /= len(X_val)
        val_losses.append(val_loss)


        # Calculo de metricas en cada epoca
        y_val_pred_norm_array = np.array(y_val_pred_norm).flatten()
        y_val_pred = y_val_pred_norm_array * (y_max - y_min) + y_min
        y_val_original = y_val * (y_max - y_min) + y_min

        rmse_val = rmse(y_val_original, y_val_pred)
        mae_val = mae(y_val_original, y_val_pred)
        r2_val = r2(y_val_original, y_val_pred)

        rmse_vals.append(rmse_val)
        mae_vals.append(mae_val)
        r2_vals.append(r2_val)

        print(f'Epoca {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
                f'RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}')
        logger.info(f'Epoca {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
                    f'RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}')
            
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(model)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping en la epoca {epoch}')
            logger.info(f'Early stopping en la epoca {epoch}')
            break

    metrics = pd.DataFrame({
        'Epoch' : range(1, len(train_losses) + 1),
        'Train Loss' : train_losses,
        'Val Loss' : val_losses,
        'Learning Rate' : learning_rates,
        'RMSE' : rmse_vals,
        'MAE' : mae_vals,
        'R2' : r2_vals
    })

    metrics.to_csv(args.metrics_file, index=False)

    return metrics, best_model
                


# plt.figure(figsize=(10, 5))
# plt.plot(range(num_ephocs), train_losses, label='Entrenamiento')
# plt.plot(range(num_ephocs), val_losses, label='Validacion')
# plt.xlabel('Epocas')
# plt.ylabel('Perdida (MSE)')
# plt.title('Curvas de perdida')
# plt.legend()
# plt.savefig('results/loss_curves.png')
# plt.show()

# plt.figure(figsize=(12, 8))
# for lambda_reg in lambda_values:
#     plt.plot(results[lambda_reg]['train_losses'], label=f'Train Loss lambda={lambda_reg}')
#     plt.plot(results[lambda_reg]['val_losses'], linestyile='--', label=f'Val Loss lambda={lambda_reg}')
# plt.xlabel('Epocas')
# plt.ylabel('Perdida (MSE)')
# plt.title('Curvas de perdida de entrenamiento y validacion para diferentes lambda')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(12, 8))
# r2_scores = [results[lambda_reg]['r2'] for lambda_reg in lambda_values]
# plt.plot(lambda_values, r2_scores, marker='o')
# plt.xscale('log')
# plt.xlabel('lambda (L2 Regularization)')
# plt.ylabel('R2')
# plt.title('R2 en validacion para diferentes lambda')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(12, 8))
# rmse_scores = [results[lambda_reg]['rmse'] for lambda_reg in lambda_values]
# plt.plot(lambda_values, rmse_scores, marker='o', color='green')
# plt.xscale('log')
# plt.xlabel('lambda (L2 Regularization)')
# plt.ylabel('RMSE')
# plt.title('RMSE en validacion para diferentes lambda')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(12, 8))
# mae_scores = [results[lambda_reg]['mae'] for lambda_reg in lambda_values]
# plt.plot(lambda_values, mae_scores, marker='o', color='red')
# plt.xscale('log')
# plt.xlabel('lambda (L2 Regularization)')
# plt.ylabel('MAE')
# plt.title('MAE en validacion para diferentes lambda')
# plt.grid(True)
# plt.show()