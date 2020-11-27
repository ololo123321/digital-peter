import os


INPUT_DIR = '/data'
OUTPUT_DIR = '/output'
HTR_MODEL_DIR = './htr'
LM_DIR = './lm'


def main():
    os.system('python input2candidates.py')
    os.system('python candidates2output.py')


if __name__ == '__main__':
    main()
