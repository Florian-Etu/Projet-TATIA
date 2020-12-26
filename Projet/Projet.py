# https://www.stat4decision.com/fr/traitement-langage-naturel-francais-tal-nlp/
# Ligne de commande: "pip install spacy" + "python -m spacy download fr_core_news_lg" + "pip install bs4" + "pip install lxml" + "pip install deep-translator"
# Lancer ce programme pour vérifier la bonne installation

import spacy
import json
import sys
import requests
from urllib.request import urlopen
from spacy.matcher import Matcher
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
from deep_translator import GoogleTranslator

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
    print("Document:\n{}\nEntities:\n{}\n\n".format(nlp, entities))
    return entities


def get_hot_words(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result


def reponse(question):
    print(question)
    hotwords = get_hot_words(question)
    createur = ["createur", "créateur", "créateure", "createure", "créatrice", "creatrice", "auteur", "auteure", "autrice", "écrit", "inventé", "inventeur", "inventeuse", "inventeure", "livre", "film"]
    if(len(hotwords)>=2):
        if(hotwords[0] in createur):
            createur = requete_dbpedia_multiple(lookup_keyword(get_hot_words(question)[1], None), "author", "name")
            if(createur == "Aucun résultat correspondant à votre recherche.\n" ):
                return requete_dbpedia_multiple(lookup_keyword(get_hot_words(question)[1], None), "creator", "name")
            return createur
    return exp_reg(nlp(question))


def exp_reg(question):
    #ANALYSE DU TYPE DE LA QUESTION (via expression régèlière en s'aidant de l'analyse NER):
    #https://spacy.io/usage/rule-based-matching
    matcher = Matcher(nlp.vocab)

    #Recherche d'une personne
    pattern = [{"LOWER": "qui"},{"POS": "AUX"}, {"ENT_TYPE": "PER"}] #QUI + AUXILIAIRE + UN NOM DE PERSONNE (éventuellement prénom + nom de famille) = ON RECHERCHE UNE PERSONNE
    matcher.add("person", None, pattern)
    
    #Recherche d'un lieu
    pattern = [{"LOWER": "où"}]
    matcher.add("Lieu", None, pattern)

    # Recherche d'une date
    pattern = [{"LOWER":{"REGEX": "année|mois|jour|date"}}] #Si la question contient année ou mois ou jour ou date = date
    pattern2 = [{"LOWER":"quand"}, {"POS": "AUX", "OP": "*"}] #Quand + auxiliaire optionnel = date
    matcher.add("Date", None, pattern, pattern2) 

    # Recherche d'un site web
    pattern = [{"LOWER":{"REGEX": "web|adresse|site|accueil|page"}}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"ENT_TYPE": "ORG"}]
    matcher.add("website", None, pattern) 

    # Recherche leader d'une ville
    pattern = [{"LOWER":{"REGEX": "maire"}}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"ENT_TYPE": "LOC"}]
    matcher.add("mayor", None, pattern)

    # Recherche leader d'un pays
    pattern = [{"LOWER":{"REGEX": "président|president|maire|chef|dirigeant|roi|renne|chancelier|chanceliere|ministre"}}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"ENT_TYPE": "LOC"}] #"maire de...", "président de la..."
    matcher.add("Leader_pays", None, pattern)
    
    matches = matcher(question)

    # Traitement selon type de la question
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        
        # span = question[start:end]  # The matched span
        # print(match_id, string_id, start, end, span.text)

        # On cherche des informations sur une personne
        if(string_id=="person"):
            return get_abstract(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="PER"][0][0], string_id)) #On execute la fonction pour faire une requete sur le nom de la personne sur laquelle on veut des informations

        elif(string_id=="mayor"):
            mayor = requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "city"), "leaderName", "populatedPlace")
            if(mayor == "Aucun résultat correspondant à votre recherche.\n" ):
                return requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "city"), string_id, "populatedPlace")
            return mayor
        
        elif(string_id=="Leader_pays"):
            return requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "country"), "leader", "populatedPlace")

        elif(string_id=="website"):
            return requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="ORG"][0][0], None), "wikiPageExternalLink", string_id)


def query(q, epr, f='application/json'):
    try:
        params = {'query': q}
        resp = requests.get(epr, params=params, headers={'Accept': f})
        return resp.text
    except Exception as e:
        print(e, file=sys.stdout)
        raise


# La fonction suivante utilise le service lookup pour rechercher à quel label correspont le mot clé entré (par
# exemple le mot clé "Macron" renverra "Emmanuel Macron")
def lookup_keyword(requete, type):
    #On traduit la requete pour la recherche avec le service lookup (qui ne prend en charge que l'anglais)
    requete_translated = GoogleTranslator(source='fr', target='en').translate(requete) 
    if(type):
        #xml_content = urlopen("https://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=" + type + "&QueryString="+requete_translated.replace(" ","%20")).read()
        xml_content = urlopen("http://akswnc7.informatik.uni-leipzig.de/lookup/api/search?label=" + requete_translated.replace(" ","%20") + "&typeName="+type).read()
    else:
        #xml_content = urlopen("https://lookup.dbpedia.org/api/search/KeywordSearch?QueryString="+requete_translated.replace(" ","%20")).read()
        xml_content = urlopen("http://akswnc7.informatik.uni-leipzig.de/lookup/api/search?label="+requete_translated.replace(" ","%20")).read()
    soup = BeautifulSoup(xml_content, "xml")
    return soup.Label.string 


def get_abstract(requete):
    json_query = json.loads(query("""prefix dbpedia: <http://dbpedia.org/resource/>
    prefix dbpedia-owl: <http://dbpedia.org/ontology/>
    SELECT DISTINCT ?abstract WHERE { 
        [ rdfs:label ?name
        ; dbpedia-owl:abstract ?abstract
        ] .
        FILTER langMatches(lang(?abstract),"fr")
        VALUES ?name {""" + '"' + requete + '"' + """@en }
        }
        LIMIT 10""","http://dbpedia.org/sparql"))

    #Dans le cas où l'on n'obtient aucun résultat en français on renvoi le résultat anglais
    if(not json_query["results"]["bindings"]):
        json_query = json.loads(query("""prefix dbpedia: <http://dbpedia.org/resource/>
        prefix dbpedia-owl: <http://dbpedia.org/ontology/>
        SELECT DISTINCT ?abstract WHERE { 
        [ rdfs:label ?name
        ; dbpedia-owl:abstract ?abstract
        ] .
        FILTER langMatches(lang(?abstract),"en")
        VALUES ?name {""" + '"' + requete + '"' + """@en }
        }
        LIMIT 10""","http://dbpedia.org/sparql"))

    if(not json_query["results"]["bindings"]):
        return "Aucun résultat correspondant à votre recherche.\n"
    
    return json_query["results"]["bindings"][0]["abstract"]["value"]+'\n'


def json_load(requete, predicate, objet):
    json_query = json.loads(query("""
        prefix dbpedia: <http://dbpedia.org/resource/>
        prefix dbpedia-owl: <http://dbpedia.org/ontology/>
        SELECT DISTINCT ?""" + predicate + """ WHERE { 
        [ rdfs:label ?""" + objet + """
        ; dbpedia-owl:""" + predicate + '?' + predicate + """] .
        VALUES ?""" + objet + '{' + '"' + requete + '"' + """@en }
        }
        LIMIT 10""", "http://dbpedia.org/sparql"))
    return json_query


def requete_dbpedia(requete, predicate, objet):
    json_query = json_load(requete, predicate, objet)

    if(not json_query["results"]["bindings"]):
        return "Aucun résultat correspondant à votre recherche.\n"

    if(objet=="website"):
        return json_query["results"]["bindings"][0][predicate]["value"] + '\n'

    json_result=json.loads(query("""SELECT ?label WHERE {<""" + json_query["results"]["bindings"][0][predicate]["value"] + """> rdfs:label ?label. FILTER langMatches(lang(?label),"en")}""", "http://dbpedia.org/sparql"))
    if(json_result["results"]["bindings"]):
        return json_result["results"]["bindings"][0]["label"]["value"]+'\n'
    else:
        return "Aucun résultat correspondant à votre recherche.\n"


def requete_dbpedia_multiple(requete, predicate, objet):
    result=""
    json_query = json_load(requete, predicate, objet)

    if(not json_query["results"]["bindings"]):
        return "Aucun résultat correspondant à votre recherche.\n"

    for resultats_requete in json_query["results"]["bindings"]:
        json_result=json.loads(query("""SELECT ?label WHERE {<""" + resultats_requete[predicate]["value"] + """> rdfs:label ?label. FILTER langMatches(lang(?label),"en")}""", "http://dbpedia.org/sparql"))
        if(json_result["results"]["bindings"]):
            result+=json_result["results"]["bindings"][0]["label"]["value"]+ " et "
    return result[:-3]+'\n'
    

if __name__ == '__main__':
    #entree = input()
    entree = "Quel est le site web de Forbes ?"
    print(reponse(entree))
    
    entree = "Qui est le créateur de Wikipedia ?"
    print(reponse(entree))

    entree = "Qui est Emmanuel Macron ?"
    print(reponse(entree))

    entree = "Qui est Nicolas Sarkozy ?"
    print(reponse(entree))

    entree = "Qui est Zinedine ?"
    print(reponse(entree))
    
    entree = "Qui est le maire de Paris ?"
    print(reponse(entree))

    entree = "Qui est le maire de New York ?"
    print(reponse(entree))

    entree = "Qui est le maire de Budapest ?"
    print(reponse(entree))

    entree = "Qui est le président des USA ?"
    print(reponse(entree))

    entree = "Qui est le président de la France ?"
    print(reponse(entree))

    entree = "Qui est la chanceliere de l'Allemagne ?"
    print(reponse(entree))

    entree = "Qui a écrit le livre Frankenstein ?"
    print(reponse(entree))

    entree = "Qui est le créateur de Goofy ?"
    print(reponse(entree))


    """doc = nlp(entree)
    #print(entree)
    PoSTagger(doc)
    print(token(doc))
    sortie = get_hot_words(entree)
    print(sortie)
    NER(doc)
    print(type_question(doc))
    type_question(nlp("Où est Emmanuel Macron ?"))
    type_question(nlp("En quelle année est né Emmanuel Macron ?"))
    print(type_question(nlp("Qui est Nicolas Sarkozy ?")))
    NER(nlp("pont de Brooklyn"))"""