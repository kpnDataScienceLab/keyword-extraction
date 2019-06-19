import pandas as pd
import ast

fname = 'keywords_ensemble.csv'
data = pd.read_csv(fname, encoding='latin')

for text, keywords in zip(data.texts, data.labels):
    print()
    print('#' * 100)
    print(f'\n\nText\n\n{text}')
    print(f'\n\nKeywords\n')
    for i, k in enumerate(ast.literal_eval(keywords)):
        print(f'{i + 1}. {k}')
    print('\n')
    input()
