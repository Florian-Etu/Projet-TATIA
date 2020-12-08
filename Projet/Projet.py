# https://www.stat4decision.com/fr/traitement-langage-naturel-francais-tal-nlp/
# Ligne de commande: "pip install spacy" + "python -m spacy download fr_core_news_sm"
# Lancer ce programme pour vérifier la bonne installation

import spacy

spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_sm")
entree = input()


def token(sentence):
    # Tokeniser la phrase
    doc = nlp(sentence)

    print("Pos tagger : ") #PoS tagger
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)

    print("\nEntité nommée") #NER
    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

    # Retourner le texte de chaque token
    print("\nText de chaque token : ")
    return [X.text for X in doc]


if __name__ == '__main__':
    print(token(entree))
