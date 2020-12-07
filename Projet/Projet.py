#https://www.stat4decision.com/fr/traitement-langage-naturel-francais-tal-nlp/
#Ligne de commande: "pip install spacy" + "python -m spacy download fr_core_news_sm"
#Lancer ce programme pour vérifier la bonne installation

import spacy

spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")
test = "Bouygues a eu une coupure de réseau à Marseille"

def token(sentence):
    # Tokeniser la phrase
    doc = nlp(sentence)
    # Retourner le texte de chaque token
    return [X.text for X in doc]

print(token(test))
