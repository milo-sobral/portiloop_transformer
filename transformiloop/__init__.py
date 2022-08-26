import logging
import os

logging.basicConfig(filename=os.path.join(os.environ['SCRATCH'], 'scripts', 'logs', f"logger_{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['$SLURM_ARRAY_TASK_ID']}.out"),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)