import datetime
import random

import logging.config
import logging


def log_name():
    start_date = datetime.datetime.now().strftime("%d%m%y_%H%M")
    random_part = random.randrange(1000, 9999, 1)
    return f'log_{start_date}_{random_part}'


def config_log():
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'./logs/{log_name()}.log')
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    log_format = u'%(filename)8s -> %(funcName)-8s [LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        handlers=[fh, ch])
