import argparse

def parse_arguments(args=None):
    """
    Analiza los argumentos de linea de comandos para el entrenamiento de una red neuronal multicapa densa (MLP).

    Esta funcion configura y parsea los argumentos necesarios para configurar el entrenamiento del MLP,
    incluyendo parametros de scheduler, regularizacion, rutas de archivos, optimizador, arquitectura de la red,
    y rutas a conjuntos de datos especificos.

    Retorna:
        argpase.Namespace: un objeto que contiene los valores de los argumentos parsados.
    """

    parser = argparse.ArgumentParser(description='Entrenamiento de una red neuronal multicapa densa (MLP)')

    # Parametros de Scheduler
    parser.add_argument('--scheduler_type', type=str, default=None,
                        choices=['linear', 'power', 'exponential'],
                        help='Tipo de learning rate scheduler (linear, power, exponential)')
    parser.add_argument('--initial_learning_rate', type=float, default=0.01,
                        help='Initial learning para entrenar la red')
    parser.add_argument('--final_lr', type=float, default=1e-7,
                        help='Learning rate final para linear decay')
    parser.add_argument('--power', type=float, default=0.5,
                        help='Power para power law decay')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='Decay rate para exponential decay')
    
    # Regularizacion L2
    parser.add_argument('--lambda_reg', type=float, default=0.0,
                        help='Hiperparametro de regularizacion L2')
    
    # Rutas de archivos
    parser.add_argument('--log_file', type=str, default='logs/training.log',
                        help='Ruta del archivo de log')
    parser.add_argument('--metrics_file', type=str, default='results/evaluation_metrics.csv',
                        help='Ruta del archivo de metricas')
    
    # Otros parametros
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Numero de epocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Tamanio del batch para MiniBatchSGD (default: tamanio completo del conjunto de entrenamiento)')
    
    # Parametros del Optimizador
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'sgd_momentum', 'adam', 'mini_batch_sgd'],
                        help='Tipo de otpimizador a utilizar')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum para SGD con momentum')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 para Adam')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 para Adam')   
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Epsilon para Adam') 
    
    # Hiperparametros de la arquitectura
    parser.add_argument('--L', type=int, default=3, help='Numero de capas en la red neuronal')
    parser.add_argument('--M_list', type=str, default='10,8,4', help='Lista de neuronas por cada capa oculta, sepradas por comas')
    
    # Argumentos para conjuntos de datos especificos
    parser.add_argument('--train_data', type=str, default='', help='Ruta al archivo de datos de entrenamiento (sin extension)')
    parser.add_argument('--val_data', type=str, default='', help='Ruta al archivo de datos de validacion (sin extension)')

    return parser.parse_args(args)