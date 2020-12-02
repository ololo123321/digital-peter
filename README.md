3rd place solution in digital peter competition: https://ods.ai/competitions/aij-petr

run submission on validation data in docker:
```bash
bash test_submission.sh
```
evaluate
```bash
python evaluate.py \
    --predictions_dir=./predictions/valid_predictions \
    --answers_dir=./data/valid_texts
```
expected output:
```bash
[OK] "сказат что вся непъриятелская си" -> "сказат что вся непъриятелская си"
[OK] "того оных от стороны" -> "того оных от стороны"
[ERR:2] "кож чинит выiскивая таких доходоф" -> "кож чинит вы iскавая таких доходоф"
[OK] "с помощию божиею кончитца" -> "с помощию божиею кончитца"
[ERR:1] "семнатцат мортироф трох пудовых" -> "семнатцат мортироф трех пудовых"
[OK] "во флотѣ для дѣйствъ про" -> "во флотѣ для дѣйствъ про"
[OK] "трову финеннъ лалантъ i там провѣдат" -> "трову финеннъ лалантъ i там провѣдат"
[OK] "правыi суд как между на" -> "правыi суд как между на"
[OK] "вышереченному ф м а с нашей сторо" -> "вышереченному ф м а с нашей сторо"
[OK] "ховѣ" -> "ховѣ"
cer: 2.792835%; wer: 14.487744%; string_acc: 59.935380%.
```
