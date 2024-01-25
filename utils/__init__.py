import logging, os
from . import options
from utils.common import time_file_str

args = options.parser.parse_args()
args.job_dir = os.path.join(args.job_dir, args.dataset, args.arch)
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

args.prefix = time_file_str()
args.job_dir = os.path.join(args.job_dir, "{}_{}_cr{}_wes{}".format(
    args.cfg, args.dist_type, args.cr, args.warmup_epochs))
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

def get_logger(file_path):
    logger = logging.getLogger('final-design-tianen')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

logger = get_logger(os.path.join(args.job_dir, "{}_{}_cr{}_wes{}_{}.log".format(
    args.cfg, args.dist_type, args.cr, args.warmup_epochs, args.prefix)))

def print_info(print_string, logger=logger):
    logger.info(print_string)
