'''All constants (strings) shared in common by multiple submodules'''

import os

GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated')

if not os.path.exists(GENERATED_DIR):
    os.mkdir(GENERATED_DIR)

if __name__ == '__main__':
    print GENERATED_DIR
