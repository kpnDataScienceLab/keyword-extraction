from rake_nltk import Rake

with open('example.txt', 'rb') as file:
    text = str(file.read())[2:-1]

print()
print(text)
print()

rake = Rake()
rake.extract_keywords_from_text(text)
keyphrases = rake.get_ranked_phrases()

for k in keyphrases:
    print(k)
