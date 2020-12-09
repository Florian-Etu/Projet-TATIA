# https://www.stat4decision.com/fr/traitement-langage-naturel-francais-tal-nlp/
# Ligne de commande: "pip install spacy" + "python -m spacy download fr_core_news_lg"
# Lancer ce programme pour vérifier la bonne installation

import spacy
import spacy
from spacy.matcher import Matcher
from collections import Counter
from string import punctuation

spacy.prefer_gpu()
nlp = spacy.load("fr_core_news_lg")

def token(doc):
    # Tokeniser la phrase
    # Retourner le texte de chaque token
    return [X.text for X in doc]

def PoSTagger(doc):
    print("Pos tagger : ") #PoS tagger
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)

    print("\nEntité nommée") #NER
    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])


def NER(nlp):
    #Retourne par exemple PER si l'entité est une personne
    entities = [(token.text, token.label_) for token in nlp.ents]
    print("Document:\n{}\nEntities:\n{}\n\n".format(doc, entities))

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

def type_question(question):
    #https://spacy.io/usage/rule-based-matching
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "qui"},{"POS": "AUX"}, {"ENT_TYPE": "PER"}] #QUI + AUXILIAIRE + UN NOM DE PERSONNE = ON RECHERCHE UNE PERSONNE
    matcher.add("Personne", None, pattern)

    pattern = [{"LOWER": "où"}]
    matcher.add("Lieu", None, pattern)

    pattern = [{"LOWER":{"REGEX": "année|mois|jour|date"}}] #Si la question contient année ou mois ou jour ou date = date
    pattern2 = [{"LOWER":"quand"}, {"POS": "AUX", "OP": "*"}] #Quand + auxiliaire optionnel = date
    matcher.add("Date", None, pattern, pattern2) 
    
    matches = matcher(question)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = question[start:end]  # The matched span
        print(match_id, string_id, start, end, span.text)

            
if __name__ == '__main__':
    #entree = input()
    entree = "Qui est Emmanuel Macron ?"
    doc = nlp(entree)
    #print(entree)
    PoSTagger(doc)
    print(token(doc))
    sortie = get_hotwords(entree)
    print(sortie)
    NER(doc)
    type_question(doc)
    type_question(nlp("Où est Emmanuel Macron ?"))
    type_question(nlp("En quelle année est né Emmanuel Macron ?"))
