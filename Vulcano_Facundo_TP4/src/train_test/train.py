import sys
import copy
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append('/home/facuvulcano/Machine-Learning/TP4/src')

from models.NN import MLP, Value
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


def train_model(args, X_train, X_val, y_train, y_val, y_min, y_max):

    logger = setup_logger('train_logger', args.log_file)

    # Definicion del scheduler
    scheduler = LearningRateScheduler(
        initial_lr=args.initial_learning_rate,
        scheduler_type=args.scheduler_type,
        final_lr=args.final_lr,
        power=args.power,
        decay_rate=args.decay_rate
    )

    # Inicializacion del modelo
    input_size = X_train.shape[1]
    layers = [args.M_list[i] for i in range (args.L)]
    print(f'\nCantidad de capas ocultas: {len(layers)}')
    for layer in range(len(layers)):
        print(f'Capa oculta: {layer + 1} tiene {layers[layer]} neuronas ')
    layers.append(1)
    model = MLP(input_size, layers)
    num_params = len(model.parameters())
    print(f'Cantidad de parametros: {num_params}')
    logger.info(f'Cantidad de parametros: {num_params}')

    if args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.initial_learning_rate)
    elif  args.optimizer == 'sgd_momentum':
        optimizer = SGDMomentum(learning_rate=args.initial_learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.initial_learning_rate, beta1=args.beta1, beta2=args.beta2)
    elif args.optimizer == 'mini_batch_sgd':
        optimizer = MiniBatchSGD(learning_rate=args.initial_learning_rate, batch_size=args.batch_size)
    else:
        raise ValueError(f"Optimizador {args.optimizer} no reconocido")

    # Determinar el tamanio del batch
    if args.batch_size is None:
        batch_size = len(X_train)
        print(f'Tamanio del batch no especificado. Usando entrenamiento batch completo: {batch_size}')
    else:
        batch_size = args.batch_size
        print(f'Tamanio del batch especificado: {batch_size}')

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

    num_batches = int(np.ceil(len(X_train) / batch_size))

    for epoch in range(1, args.num_epochs + 1):
        current_lr = scheduler.get_lr(epoch, args.num_epochs)
        learning_rates.append(current_lr)

        optimizer.lr = current_lr

        epoch_loss = 0.0

        # Entrenamiento por batches
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_train.iloc[start:end]
            y_batch = y_train[start:end]

            model.zero_grad()

            batch_loss = 0.0

            for i in range(len(X_batch)):
                x = X_batch.iloc[i].values
                y_true = y_batch[i]
                y_pred = model(x)

                # Calcular la perdida total
                loss = total_loss(y_true, y_pred, model, args.lambda_reg)
                batch_loss += loss.data

                # Backpropagation
                loss.backward()

            # Obtener parametros y gradientes
            for param in model.parameters():
                param.grad /= len(X_batch)

            params = model.parameters()
            grads = [p.grad for p in params]

            optimizer.update(params, grads)

            epoch_loss += batch_loss

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

        # print(f'Epoca {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
        #         f'RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}')
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
            print(f'\nEarly stopping en la epoca {epoch}')
            print(f'Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
                f'RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}')
            logger.info(f'Early stopping en la epoca {epoch}')
            break 
        
    print(f'\nEntrenamiento durante {epoch} epocas.')
    print(f'Resultados finales:')
    print(f'Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, '
                f'RMSE={rmse_vals[-1]:.4f}, MAE={mae_vals[-1]:.4f}, R2={r2_vals[-1]:.4f}')

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
                


def train_final_model(args, X_train, y_train):

    logger = setup_logger('final_train_logger', args.log_file)

    # Definicion del scheduler
    scheduler = LearningRateScheduler(
        initial_lr=args.initial_learning_rate,
        scheduler_type=args.scheduler_type,
        final_lr=args.final_lr,
        power=args.power,
        decay_rate=args.decay_rate
    )

    # Inicializacion del modelo
    input_size = X_train.shape[1]
    layers = [int(neurons) for neurons in args.M_list.split(',')]
    print(f'\nCantidad de capas ocultas: {len(layers)}')
    for layer in range(len(layers)):
        print(f'Capa oculta: {layer + 1} tiene {layers[layer]} neuronas ')
    layers.append(1)
    model = MLP(input_size, layers)
    num_params = len(model.parameters())
    print(f'Cantidad de parametros: {num_params}')
    logger.info(f'Cantidad de parametros: {num_params}')

    if args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.initial_learning_rate)
    elif  args.optimizer == 'sgd_momentum':
        optimizer = SGDMomentum(learning_rate=args.initial_learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.initial_learning_rate, beta1=args.beta1, beta2=args.beta2)
    elif args.optimizer == 'mini_batch_sgd':
        optimizer = MiniBatchSGD(learning_rate=args.initial_learning_rate, batch_size=args.batch_size)
    else:
        raise ValueError(f"Optimizador {args.optimizer} no reconocido")

    # Determinar el tamanio del batch
    if args.batch_size is None:
        batch_size = len(X_train)
        print(f'Tamanio del batch no especificado. Usando entrenamiento batch completo: {batch_size}')
    else:
        batch_size = args.batch_size
        print(f'Tamanio del batch especificado: {batch_size}')

    # Inicializacion de listas para almacenar perdidas y metricas
    train_losses = []
    learning_rates = []
    rmse_train_vals = []
    mae_train_vals = []
    r2_train_vals = []

    # Early Stopping
    patience = 50
    patience_counter = 0

    num_batches = int(np.ceil(len(X_train) / batch_size))

    for epoch in range(1, args.num_epochs + 1):
        current_lr = scheduler.get_lr(epoch, args.num_epochs)
        learning_rates.append(current_lr)
        optimizer.lr = current_lr

        epoch_loss = 0.0
        all_preds = []
        all_trues = []

        # Entrenamiento por batches
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_train.iloc[start:end]
            y_batch = y_train[start:end]

            model.zero_grad()

            batch_loss = 0.0

            for i in range(len(X_batch)):
                x = X_batch.iloc[i].values
                y_true = y_batch[i]
                y_pred = model(x)
                #print(f'Prediccion: {y_pred}, Verdadero: {y_true}')

                all_preds.append(y_pred)
                all_trues.append(y_true)

                # Calcular la perdida total
                loss = total_loss(y_true, y_pred, model, args.lambda_reg)
                batch_loss += loss.data

                # Backpropagation
                loss.backward()

            # Obtener parametros y gradientes
            for param in model.parameters():
                param.grad /= len(X_batch)

            params = model.parameters()
            grads = [p.grad for p in params]
            optimizer.update(params, grads)
            epoch_loss += batch_loss

        # Calcular perdida promedio de entrenamiento
        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)

        # Convertir predicciones (Value) a float antes de calcular las métricas
        all_preds = np.array([pred.data if isinstance(pred, Value) else pred for pred in all_preds])
        all_trues = np.array([true.data if isinstance(true, Value) else true for true in all_trues])


        rmse_train = rmse(all_trues, all_preds)
        mae_train = mae(all_trues, all_preds)
        r2_train = r2(all_trues, all_preds)

        rmse_train_vals.append(rmse_train)
        mae_train_vals.append(mae_train)
        r2_train_vals.append(r2_train)

        logger.info(f'Epoca {epoch}: Train Loss={train_loss:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train}, R2={r2_train:.4f}')
            
        if patience_counter >= patience:
            print(f'\nEarly stopping en la epoca {epoch}')
            print(f'Train Loss={train_loss:.4f}')
            logger.info(f'Early stopping en la epoca {epoch}')
            break 
        
    print(f'\nEntrenamiento durante {epoch} epocas.')
    print(f'Resultados finales:')
    print(f'Train Loss={train_losses[-1]:.4f}, RMSE={rmse_train_vals[-1]:.4f}, MAE={mae_train_vals[-1]:.4f}, R2={r2_train_vals[-1]:.4f}')

    metrics = pd.DataFrame({
        'Epoch' : range(1, len(train_losses) + 1),
        'Train Loss' : train_losses,
        'Learning Rate' : learning_rates,
        'Rmse Train' : rmse_train_vals,
        'Mae Train' : mae_train_vals,
        'R2 Train' : r2_train_vals
    })

    metrics.to_csv(args.metrics_file, index=False)

    return metrics, model