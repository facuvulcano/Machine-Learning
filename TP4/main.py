import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.utils.parse import parse_arguments
from src.datasets.load_data import load_and_prepare_data
from src.train_test.train import train_model



def main():
    
    print("Iniciando el script")

    # Parsear los argumentos de linea de comandos
    args = parse_arguments()

    
    # Crear directorios necesaruis si no existen

    log_dir = os.path.dirname(args.log_file)
    metrics_dir = os.path.dirname(args.metrics_file)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Cargar y preparar los datos
    X_train, X_val, y_train, y_val, X_test, y_test, y_min, y_max = load_and_prepare_data()

    # Entrenar el modelo
    metrics, best_model = train_model(args, X_train, X_val, y_train, y_val, y_min, y_max)

    # Guardar las metricas de evaluacion (ultima epoca)
    final_metrics = metrics.tail(1)
    final_metrics.to_csv(args.metrics_file, mode='a', header=False, index=False)

    # Imprimir resumen
    print("\nEntrenamiento completado.")
    print(f"Metricas guardads en: {args.metrics_file}")

if __name__ == "__main__":
    main()