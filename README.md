3rd place solution in digital peter competition: https://ods.ai/competitions/aij-petr

setup:
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```
run submission on validation data:
```bash
bash test_submission.sh
```
evaluate
```bash
python evaluate.py \
    --predictions_dir=./predictions/valid_predictions \
    --answers_dir=./data/valid_texts
```
