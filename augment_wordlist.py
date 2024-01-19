import json
import collections
import re

with open('SAOL.txt', 'r', encoding='utf8') as file:
    text = file.readlines()
    text = [re.sub(r'\n', '', word) for word in text]
    text = [re.sub(r'é', 'e', word).upper() for word in text]
    text = [word for word in text if "-" not in word]
    WORDLIST = set(text)

lookup_lemmas = collections.defaultdict(list)
with open('sv_lemma_lookup.json', 'rb+') as f:
    data = json.load(f)
    for key, val in data.items():
        lookup_lemmas[val].append(key)

no_lemmas = 0
new_wordlist = set()
for word in WORDLIST:
    new_wordlist.add(word.lower())
    if word.lower() in lookup_lemmas:
        for lemma in lookup_lemmas[word.lower()]:
            new_wordlist.add(lemma)
new_wordlist = list(new_wordlist)
new_wordlist = [word+"\n" for word in new_wordlist]
new_wordlist = [word for word in new_wordlist if not "-" in word and not ":" in word]
new_file = open("SAOL_AUGMENTED.txt", 'w+')
new_file.writelines(new_wordlist)