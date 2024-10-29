import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """
    Configura y devuelve un logger para registrar eventos en un archivo de log.

    Este logger permite registrar mensajes de log en un archivo especifico, con un formato estandar
    que incluye la fecha, el nivel del log, y el mensaje. Si la ruta del archivo de log no existe,
    se crea automaticamente.

    Args:
        name (str): Nombre del logger.
        log_file (str): Ruta completa del archivo de log donde se almacenaran los mensajes.
        level (int, opcional): Nivel de logging, que define el nivel de mensaje que se registraran.
                               El valor por defecto es 'logging.INFO', que registra mensajes de nivel
                               INFO y superiores.

    Returns:
        logging.Logger: El logger configurado.
    """

    log_dir = os.path.dirname(log_file)

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger