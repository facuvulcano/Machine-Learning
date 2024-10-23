import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.utils.parse import parse_arguments
from src.datasets.load_data import load_and_prepare_data
from src.train_test.train import train_model



def main():

    # Parsear los argumentos de linea de comandos
    args = parse_arguments()

    # Convertir M_list de cadena a lista de enteros
    if isinstance(args.M_list, str):
        try:
            args.M_list = [int(m) for m in args.M_list.split(',')]
        except ValueError:
            raise ValueError("El argumento --M_list debe ser una lista de enteros separados por comas, por ejemplo: '20,15,10")

    if len(args.M_list) != args.L:
        raise ValueError(f"La longitud de M_list ({len(args.M_list)}) no coincide con L ({args.L})")
    # Crear directorios necesaruis si no existen

    log_dir = os.path.dirname(args.log_file)
    metrics_dir = os.path.dirname(args.metrics_file)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Cargar y preparar los datos
    X_train, X_val, y_train, y_val, X_test, y_test, y_min, y_max = load_and_prepare_data(
        train_data_path=args.train_data,
        val_data_path=args.val_data
    )

    # Entrenar el modelo
    metrics, best_model = train_model(args, X_train, X_val, y_train, y_val, y_min, y_max)

    metrics.to_csv(args.metrics_file, mode='a', header=False, index=False)

    # Imprimir resumen
    print("\nEntrenamiento completado.")
    print(f"Metricas guardads en: {args.metrics_file}")

if __name__ == "__main__":
    main()