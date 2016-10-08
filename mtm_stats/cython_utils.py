import os
import shutil
import numpy as np
import Cpyx
from constants import GENERATED_DIR

FORCE_RECOMPILE = False

from generated import cy_mtm_stats

def compile_core(target_directory=GENERATED_DIR):
    cwd = os.getcwd()
    if target_directory is not None:
        for f in ['cy_mtm_stats.pyx', 'mtm_stats_core.c', 'mtm_stats_core.h',
                  'c_python.pxd', 'c_numpy.pxd']:
            shutil.copy(f, os.path.join(target_directory, f))
        with open(os.path.join(target_directory, '__init__.py'), 'w') as fid:
            fid.write('')
        os.chdir(target_directory)
    Cpyx.cpyx('cy_mtm_stats.pyx', 'mtm_stats_core.c', gcc_options=['-fPIC', '-fopenmp'],
              ld_options=['-fopenmp'], use_distutils=True)
    
    os.chdir(cwd)

try:
    if FORCE_RECOMPILE:
        raise Exception('Import skipped until after forced compilation')
    from generated import cy_mtm_stats
except Exception, ImportError:
    compile_core()
    from generated import cy_mtm_stats

def run_mtm_stats(sba_list, chunk_length, cutoff=0):
    all_sparse_counts = cy_mtm_stats.cy_mtm_stats(sba_list, chunk_length, cutoff=cutoff)
    return all_sparse_counts

if __name__ == '__main__':
    compile_core()
    from generated import cy_mtm_stats
