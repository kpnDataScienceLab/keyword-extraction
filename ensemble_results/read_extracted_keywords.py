import pandas as pd
import ast

print("Score type: ", end='')
score_type = input()
fname = f'keywords_ensemble_{score_type}_score.csv'
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
