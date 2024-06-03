import os
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized

def set_up_env(random_state=42):
    torch.manual_seed(random_state)
    if not torch.distributed.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)
    