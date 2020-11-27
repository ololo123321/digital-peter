import os
import sys
import re


# def predict(text):
#     text = ' '.join(list(text.replace(' ', '|')))
#     cmd = './kenlm/build/bin/query ./kenlm/build/bin/model.binary'
#     res = os.popen(f'echo "{text}" | {cmd}').read()
#     score = re.search('Total: (\-\d+\.\d+)', res).group(1)
#     score = float(score)
#     return score


if __name__ == '__main__':
    # print(predict('мама мыла раму'))
    # sys.path.insert(0, './kenlm/python')
    sys.path = ['./kenlm/python']
    print(os.listdir('./kenlm/python'))
    import kenlm
    print(vars(kenlm))
