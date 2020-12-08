# https://www.stat4decision.com/fr/traitement-langage-naturel-francais-tal-nlp/
# Ligne de commande: "pip install spacy" + "python -m spacy download fr_core_news_sm"
# Lancer ce programme pour vérifier la bonne installation

import spacy
import spacy
from collections import Counter
from string import punctuation

spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")

def token(sentence):
    # Tokeniser la phrase
    doc = nlp(sentence)
    # Retourner le texte de chaque token
    return [X.text for X in doc]

def PoSTagger(sentence):
    doc = nlp(sentence)
    print("Pos tagger : ") #PoS tagger
    for token in nlp(sentence):
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)

    print("\nEntité nommée") #NER
    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

if __name__ == '__main__':
    entree = input()
    #print(entree)
    #print(token(entree))
    sortie = get_hotwords(entree)
    print(PoSTagger(entree))
    print(sortie)
