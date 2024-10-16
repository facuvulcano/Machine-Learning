import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Entrenamiento de una red neuronal multicapa densa (MLP)')

    # Parametros de Scheduler
    parser.add_argument('--scheduler_type', type=str, default='linear',
                        choices=['linear', 'power', 'exponential'],
                        help='Tipo de learning rate scheduler (linear, power, exponential)')
    parser.add_argument('--final_lr', type=float, default=1e-7,
                        help='Learning rate final para linear decay')
    parser.add_argument('--power', type=float, default=0.5,
                        help='Power para power law decay')
    parser.add_argument('--decay_rate', type=float, default=0.01,
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamanio del batch para entrenamiento')
    return parser.parse_args()