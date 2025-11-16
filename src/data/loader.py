import pandas as pd

COLS = ['engine_id','cycle'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]

def load_fd_subset(root: str, subset: str='FD001'):
    train = pd.read_csv(f'{root}/train_{subset}.txt', sep='\s+', header=None)
    test  = pd.read_csv(f'{root}/test_{subset}.txt',  sep='\s+', header=None)
    rul   = pd.read_csv(f'{root}/RUL_{subset}.txt',   sep='\s+', header=None)
    # Some mirrors have trailing spaces -> auto sep handles it; set columns
    train.columns = COLS
    test.columns = COLS
    return train, test, rul
