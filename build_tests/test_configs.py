import sys
sys.path.append('.')
from libs.lib import *
from configs.options import *

def main():

    # reading classes and territories
    classes = read_csv_file(classes_path)
    territories = read_csv_file(territories_path)

    # test if all queried classes are in the options:


if __name__ == '__main__':
    main()
