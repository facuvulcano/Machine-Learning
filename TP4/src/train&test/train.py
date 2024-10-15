import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/home/facuvulcano/Machine-Learning/TP4/src')
from datasets.cross_validation import cross_val
from datasets.train_validation import train_val_split
from datasets.preprocesing import processing, normalize
from models.NN import MLP, Value
from metrics.rmse import rmse
from metrics.mae import mae
from metrics.r2 import r2
from utils.logger import setup_logger
from schedulers.learning_rate_schedulers import LearningRateScheduler



# CARGA Y PREPROCESAMIENTO DE DATOS

df = pd.read_csv('/home/facuvulcano/Machine-Learning/TP4/data/raw/toyota_dev.csv')
test_df = pd.read_csv('/home/facuvulcano/Machine-Learning/TP4/data/raw/toyota_test.csv')
df_path = processing(df)
test_df_path = processing(test_df)
processed_df = pd.read_csv(df_path)
processed_test_df = pd.read_csv(test_df_path)

X_test = processed_test_df.drop(columns=['Precio'])
y_test = processed_test_df['Precio']

X_train, X_val, y_train, y_val = train_val_split(processed_df, 'Precio')

features_to_normalize = ['Año','Kilómetros']
min_max_values = {feature : (X_train[feature].min(), X_train[feature].max()) for feature in features_to_normalize}


for feature in features_to_normalize:
    min_val, max_val = min_max_values[feature]
    X_train[feature] = normalize(X_train, feature, min_val, max_val, [0, 1])
    X_val[feature] = normalize(X_val, feature, min_val, max_val, [0, 1])
    X_test[feature] = normalize(X_test, feature, min_val, max_val, [0, 1])

y_min = y_train.min()
y_max = y_train.max()

y_train_norm = (y_train - y_min) / (y_max - y_min)
y_val_norm = (y_val - y_min) / (y_max - y_min)
y_test_norm = (y_test - y_min) / (y_max - y_min)

#-----------------------------------------------------------------------


# DEFINICION DE FUNCIONES DE PERDIDA Y REGULARIZACION
def mse_loss(y_true, y_pred):
    return (y_pred - y_true)**2

def l2_regularization(model, lambda_reg):
    l2_sum = 0.0
    for param in model.parameters():
        if hasattr(param, 'data'):
            l2_sum += param.data ** 2
    return lambda_reg * l2_sum

def total_loss(y_true, y_pred, model, lambda_reg):
    return mse_loss(y_true, y_pred) + l2_regularization(model, lambda_reg)

#------------------------------------------------------------------------------

initial_learning_rate = 1e-5
num_ephocs = 1000
batch_size = 32

X_train = np.array(X_train)
X_val = np.array(X_val)

logger = setup_logger('train_logger', 'logs/training.log')
lambda_values = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
results = {}

for lambda_reg in lambda_values:
    print(f'\nEntrenando con lambda = {lambda_reg}')

    model = MLP(19, [10, 8, 4, 1])
    print(f'Cantidad de parametros: {len(model.parameters())}')

    train_losses = []
    val_losses = []
    learning_rates = []

    # Selection of scheduler type
    scheduler = LearningRateScheduler(
        initial_lr=initial_learning_rate,
        scheduler_type='linear',
        final_lr=1e-7,
        power=0.5,
        decay_rate=0.01
    )

    # Early stopping   
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    for epoch in range(num_ephocs):
        current_lr = scheduler.get_lr(epoch, num_ephocs)
        learning_rates.append(current_lr)
        epoch_loss = 0

        for i in range(len(X_train)):
            x = X_train[i]
            y_true = y_train_norm.iloc[i]
            y_pred = model(x)
            
            #calculo de la perdida (MSE)
            loss = total_loss(y_true, y_pred, model, lambda_reg)
            epoch_loss += loss.data

            #Backpropagation
            loss.backward()

            #Acumular gradientes
            for p in model.parameters():
                p.data += -current_lr * p.grad
            
            model.zero_grad()
        
        train_loss = epoch_loss / len(X_train)
        train_losses.append(train_loss)

        val_loss = 0
        y_val_pred_norm = []
        for i in range(len(X_val)):
            x = X_val[i]
            y_true = y_val_norm.iloc[i]
            y_pred_norm = model(x)
            y_val_pred_norm.append(y_pred_norm.data)
            val_loss += (y_pred_norm.data - y_true) ** 2
        val_loss /= len(X_val)
        val_losses.append(val_loss)

        if epoch % 100 == 0:
            y_val_pred_norm_array = np.array(y_val_pred_norm).flatten()
            y_val_pred = y_val_pred_norm_array * (y_max - y_min) + y_min
            y_val_original = y_val.values

            rmse_val = rmse(np.array(y_val_original), y_val_pred)
            mae_val = mae(np.array(y_val_original), y_val_pred)
            r2_val = r2(np.array(y_val_original), y_val_pred)
            print(f'Epoca {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping on epoch: {epoch}')
            break

        logger.info(f'Epoca {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
    
    results[lambda_reg] = {
        'train_losses' : train_losses,
        'val_losses' : val_losses,
        'learning_rates' : learning_rates,
        'rmse' : rmse_val,
        'mae' : mae_val,
        'r2' : r2_val
    }

metrics_df = pd.DataFrame({
    'Epoch' : range(num_ephocs),
    'Train Loss' : train_losses,
    'Val Loss' : val_losses,
    'RMSE'  : rmse_val,
    'MAE'   : mae_val,
    'R2' : r2_val
})

metrics_df.to_csv('results/evaluation_metrics.csv', index=False)


plt.figure(figsize=(10, 5))
plt.plot(range(num_ephocs), train_losses, label='Entrenamiento')
plt.plot(range(num_ephocs), val_losses, label='Validacion')
plt.xlabel('Epocas')
plt.ylabel('Perdida (MSE)')
plt.title('Curvas de perdida')
plt.legend()
plt.savefig('results/loss_curves.png')
plt.show()

plt.figure(figsize=(12, 8))
for lambda_reg in lambda_values:
    plt.plot(results[lambda_reg]['train_losses'], label=f'Train Loss lambda={lambda_reg}')
    plt.plot(results[lambda_reg]['val_losses'], linestyile='--', label=f'Val Loss lambda={lambda_reg}')
plt.xlabel('Epocas')
plt.ylabel('Perdida (MSE)')
plt.title('Curvas de perdida de entrenamiento y validacion para diferentes lambda')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
r2_scores = [results[lambda_reg]['r2'] for lambda_reg in lambda_values]
plt.plot(lambda_values, r2_scores, marker='o')
plt.xscale('log')
plt.xlabel('lambda (L2 Regularization)')
plt.ylabel('R2')
plt.title('R2 en validacion para diferentes lambda')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
rmse_scores = [results[lambda_reg]['rmse'] for lambda_reg in lambda_values]
plt.plot(lambda_values, rmse_scores, marker='o', color='green')
plt.xscale('log')
plt.xlabel('lambda (L2 Regularization)')
plt.ylabel('RMSE')
plt.title('RMSE en validacion para diferentes lambda')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
mae_scores = [results[lambda_reg]['mae'] for lambda_reg in lambda_values]
plt.plot(lambda_values, mae_scores, marker='o', color='red')
plt.xscale('log')
plt.xlabel('lambda (L2 Regularization)')
plt.ylabel('MAE')
plt.title('MAE en validacion para diferentes lambda')
plt.grid(True)
plt.show()