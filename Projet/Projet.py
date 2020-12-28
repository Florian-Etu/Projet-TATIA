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


def reponse(question):
    print(question)
    question = question.replace("l'", "l' ")
    hotwords = get_hotwords(question)
    alliance = ["epoux", "époux", "epouse", "épouse", "mari", "femme", "mariée", "marié", "marier", "mariage", "partnaire", "relation", "union"]
    prog = ["language", "langage", "programmation", "languages", "langages", "programmations", "programation", "program"]
    capital = ["capitale", "capital"]
    monnaie = ["devise", "devises", "monnaie", "monnaies"]

    if(len(hotwords)>=2):

        # Recherche de la capital d'un pays
        if (any(item in hotwords for item in capital) or ("grande" in hotwords and "ville" in hotwords)):
            capital = requete_dbpedia(lookup_keyword(hotwords[-1], "country"), "capital")
            return capital

        # Recherche partenaire d'une personne
        if(any(item in hotwords for item in alliance)):
            partenaire = requete_dbpedia(lookup_keyword(hotwords[-1], "person"), "spouse")
            if(partenaire == "Aucun résultat correspondant à votre recherche.\n" ):
                partenaire = requete_dbpedia(lookup_keyword(hotwords[-1], "person"), "partner")
                if(partenaire == "Aucun résultat correspondant à votre recherche.\n" ):
                    partenaire = requete_dbpedia(lookup_keyword(hotwords[-1], "person"), "wife", "dbp")
                    if(partenaire == "Aucun résultat correspondant à votre recherche.\n" ):
                        partenaire = requete_dbpedia(lookup_keyword(hotwords[-1], "person"), "husband", "dbp")
                    if(partenaire == "Aucun résultat correspondant à votre recherche.\n" ):
                        partenaire = requete_dbpedia(lookup_keyword(hotwords[-1], "person"), "union", "dbp")
                    if(partenaire == "Aucun résultat correspondant à votre recherche.\n" ):
                        return requete_dbpedia(lookup_keyword(hotwords[-1], "person"), "relationship", "dbp")
            return partenaire

        # Recherche language de programmation d'un logiciel
        if(any(item in hotwords for item in prog)):
            return requete_dbpedia_multiple(lookup_keyword(hotwords[-1], "software"), "programmingLanguage")

        # Recherche devise / monnaie d'un pays
        if(any(item in hotwords for item in monnaie)):
            return requete_dbpedia_multiple(lookup_keyword(hotwords[-1], "country"), "currency")

    return exp_reg(nlp(question))


def exp_reg(question):
    #ANALYSE DU TYPE DE LA QUESTION (via expression régèlière en s'aidant de l'analyse NER):
    #https://spacy.io/usage/rule-based-matching
    matcher = Matcher(nlp.vocab)

    # Recherche d'une date
    pattern = [{"LOWER":{"REGEX": "années?|mois|jours?|dates?"}}] #Si la question contient année ou mois ou jour ou date = date
    pattern2 = [{"LOWER":"quand"}, {"POS": "AUX", "OP": "*"}] #Quand + auxiliaire optionnel = date
    matcher.add("Date", None, pattern, pattern2) 

    # Recherche d'un site web
    pattern = [{"LOWER":{"REGEX": "web|add?resse|site|accueil|page"}}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "*"}, {"IS_PUNCT": True, "OP": "*"}, {"ENT_TYPE": "ORG"}]
    matcher.add("website", None, pattern) 

    # Recherche leader d'une ville
    pattern = [{"LOWER":{"REGEX": "maire"}}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "*"}, {"IS_PUNCT": True, "OP": "*"}, {"ENT_TYPE": "LOC"}]
    matcher.add("mayor", None, pattern)

    # Recherche leader d'un pays
    pattern = [{"LOWER":{"REGEX": "pr[ée]sidents?|maires?|chefs?|dirigeant|roi|renne|chancelier|chanceli[eè]re|ministre"}}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "*"}, {"IS_PUNCT": True, "OP": "*"}, {"ENT_TYPE": "LOC"}]  # "maire de...", "président de la..."
    matcher.add("Leader_pays", None, pattern)

    # Recherche voisins
    pattern = [{"LOWER":{"REGEX": "voisins?|fronti[èeé]res?|autours?"}}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "*"}, {"IS_PUNCT": True, "OP": "*"},{"ENT_TYPE": "LOC"}] 
    matcher.add("voisin", None, pattern)

    # Recherche dans quel pays se trouve une ville ou un lieu
    pattern = [{"LOWER": {"REGEX": "où"}}, {"POS": "AUX", "OP": "*"}, {"POS": "PRON", "OP": "*"}, {"POS": "VERB","OP": "*"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "*"}, {"POS": "ADP", "OP": "*"}, {"POS": "NOUN", "OP": "*"}, {"IS_PUNCT": True, "OP": "*"}, {"ENT_TYPE": "LOC"}]  # Où est ce que se trouve la ville de nomVille ? et "Où est nomVille ?"
    pattern2 = [{"LOWER": {"REGEX": "dans"}}, {"POS": "ADJ", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "*"}, {"POS": "PRON", "OP": "*"}, {"POS": "VERB","OP": "*"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "*"}, {"POS": "ADP", "OP": "*"}, {"IS_PUNCT": True, "OP": "*"}, {"ENT_TYPE": "LOC"}]  # "Dans quel pays se trouve la ville de nomVille? "
    matcher.add("pays", None, pattern, pattern2)

    # Recherche langues pays
    pattern = [{"LOWER": {"REGEX": "langues?|parl[ée]r?e?s?"}}, {"OP": "*"}, {"ENT_TYPE": "LOC"}]
    matcher.add("langue", None, pattern)

    # Recherche d'une appartenance / possession
    pattern = [{"LOWER": {"REGEX": "poss[èe]den?t?|app?artienn?e?n?t|app?artenance|possess?ions?"}}, {"OP": "*"}, {"ENT_TYPE": "MISC"}]
    pattern2 = [{"LOWER": {"REGEX": "poss[èe]den?t?|app?artienn?e?n?t|app?artenance|possess?ions?"}}, {"OP": "*"}, {"ENT_TYPE": "ORG"}]
    matcher.add("appartenance", None, pattern, pattern2)

    # Recherche gagnant de prix
    pattern = [{"LOWER": {"REGEX": "gagn[ée]e?s?r?"}}, {"OP": "*"}, {"ENT_TYPE": "ORG"}]
    matcher.add("awards", None, pattern)

    # Recherche d'un créateur / auteur / développeur
    pattern = [{"LOWER": {"REGEX": "cr[ée]ateure?s?|cr[ée]atrice|auteure?|autrice|[ée]crite?|invent[ée]e?|inventeure?|inventeuse|livres?|d[ée]velop?pée?|d[ée]velop?per|d[ée]veloppeur?s?e?s?|cr[ée]é?e?r?|produits?|jeux?|videos?|vid[ée]os?"}}, {"OP": "*"}, {"ENT_TYPE": "MISC"}]
    matcher.add("createur", None, pattern, pattern2)

    # Recherche de concepteur / designer
    pattern = [{"LOWER": {"REGEX": "con[çc]ue?s?|design[ée]e?s?|dessin[ée]e?r?s?|construit?e?s?|constructions?"}}, {"POS": "DET", "OP": "*"}, {"ENT_TYPE": "LOC"}]
    matcher.add("concepteur", None, pattern)

    # Recherche d'acteurs
    pattern = [{"LOWER": {"REGEX": "acteure?s?|actrices?"}}, {"OP": "*"}, {"ENT_TYPE": "MISC"}]
    matcher.add("acteur", None, pattern)

    # Recherche cause de décès
    pattern = [{"LOWER": {"REGEX": "d[ée]c[èée]s?|d[ée]c[ée]d[eé]e?r?|morte?s?|mourire?"}}, {"OP": "*"}, {"ENT_TYPE": "PER"}]
    matcher.add("deces", None, pattern)
    
    # Recherche d'une personne
    pattern = [{"LOWER": "qui"},{"POS": "AUX"}, {"ENT_TYPE": "PER"}] #QUI + AUXILIAIRE + UN NOM DE PERSONNE (éventuellement prénom + nom de famille) = ON RECHERCHE UNE PERSONNE
    matcher.add("person", None, pattern)

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
            mayor = requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "city"), "leaderName")
            if(mayor == "Aucun résultat correspondant à votre recherche.\n" ):
                mayor =  requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "city"), string_id)
                if(mayor == "Aucun résultat correspondant à votre recherche.\n"):
                    mayor =  requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "settlement"), string_id)
                    if(mayor == "Aucun résultat correspondant à votre recherche.\n"):
                        return requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "settlement", False), string_id)                            
            return mayor
        
        elif(string_id=="Leader_pays"):
            return requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "country"), "leader")

        elif(string_id=="website"):
            return requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="ORG"][0][0], None), "wikiPageExternalLink")

        elif(string_id == "pays"):
            if("lac" in question.text):
                return requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_ == "LOC"][0][0], "lake"), "country")

            plusieurs = ["quelles","quels","sont","quel"] #quel peut etre utiliser pour representer plusieurs pays(dans quel pays se trouve ... :le résultat peut etre plusieurs pays)
            existance = ["commence", "source", "origine"]
            if (any(item in question.text for item in existance)):
                return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_ == "LOC"][0][0], None),"sourceCountry")
            elif(any(item in question.text for item in plusieurs)):
                return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_ == "LOC"][0][0],None), "country")
            else:
                return requete_dbpedia(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_ == "LOC"][0][0], None), "country")

        elif(string_id=="voisin"):
            return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], None), "borderingstates", "dbp")
        
        elif(string_id=="createur"):
            auteur = requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="MISC"][0][0], None), "developer")
            if(auteur == "Aucun résultat correspondant à votre recherche.\n" ):
                auteur = requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="MISC"][0][0], None), "author")
                if(auteur == "Aucun résultat correspondant à votre recherche.\n" ):
                    return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="MISC"][0][0], None), "creator")
            return auteur

        elif(string_id=="appartenance"):
            recherche = [(ent.text, ent.label_) for ent in question.ents if ent.label_=="ORG"]
            if(len(recherche)<1):
                recherche = [(ent.text, ent.label_) for ent in question.ents if ent.label_=="MISC"]
            recherche=lookup_keyword(recherche[0][0], None)
            fondateur = requete_dbpedia_multiple(recherche, "owner")
            if(fondateur == "Aucun résultat correspondant à votre recherche.\n" ):
                fondateur = requete_dbpedia_multiple(recherche, "founders", "dbp")
                if(fondateur == "Aucun résultat correspondant à votre recherche.\n" ):
                    fondateur = requete_dbpedia_multiple(recherche, "keyPerson")
                    if(fondateur == "Aucun résultat correspondant à votre recherche.\n" ):        
                        return requete_dbpedia_multiple(recherche, "keyPeople", "dbp")
            return fondateur

        elif(string_id=="awards"):
            return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="ORG"][0][0], "work"), string_id, "dbp")

        elif(string_id=="acteur"):
            return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="MISC"][0][0], None), "starring")

        elif(string_id=="deces"):
            return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="PER"][0][0], "person"), "deathCause")

        elif(string_id=="langue"):
            return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], "country"), "officialLanguage")
        
        elif(string_id=="concepteur"):
            concepteur = requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], None), "designer", "dbp")
            if(concepteur == "Aucun résultat correspondant à votre recherche.\n" ):
                return requete_dbpedia_multiple(lookup_keyword([(ent.text, ent.label_) for ent in question.ents if ent.label_=="LOC"][0][0], None), "architect")
            return concepteur



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
def lookup_keyword(requete, type, translate=True):
    #On traduit la requete pour la recherche avec le service lookup (qui ne prend en charge que l'anglais)
    requete_translated = requete
    if(translate):
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


def json_load(requete, predicate, entity_of_type):
    json_query = json.loads(query("""
        prefix dbpedia: <http://dbpedia.org/resource/>
        prefix dbpedia-owl: <http://dbpedia.org/ontology/>
        SELECT DISTINCT ?""" + predicate + """ WHERE { 
        [ rdfs:label ?s
        ; dbpedia-owl:""" + predicate + '?' + predicate + """] .
        VALUES ?s{""" + '"' + requete + '"' + """@en }
        }
        LIMIT 10""", "http://dbpedia.org/sparql"))

    if(entity_of_type == "dbp"):
        json_query = json.loads(query("""
        prefix dbpedia: <http://dbpedia.org/resource/>
        prefix dbp: <http://dbpedia.org/property/>
        SELECT DISTINCT ?""" + predicate + """ WHERE { 
        [ rdfs:label ?s
        ; dbp:""" + predicate + '?' + predicate + """] .
        VALUES ?s{""" + '"' + requete + '"' + """@en }
        }
        LIMIT 10""", "http://dbpedia.org/sparql"))
    return json_query


def requete_dbpedia(requete, predicate, entity_of_type="dbo"):
    json_query = json_load(requete, predicate, entity_of_type)

    if(not json_query["results"]["bindings"]):
        return "Aucun résultat correspondant à votre recherche.\n"

    if(not json_query["results"]["bindings"][0][predicate]["value"].startswith("http://dbpedia.org")):
        return json_query["results"]["bindings"][0][predicate]["value"] + '\n'

    json_result=json.loads(query("""SELECT ?label WHERE {<""" + json_query["results"]["bindings"][0][predicate]["value"] + """> rdfs:label ?label. FILTER langMatches(lang(?label),"fr")}""", "http://dbpedia.org/sparql"))
    if(json_result["results"]["bindings"]):
        return json_result["results"]["bindings"][0]["label"]["value"]+'\n'
    else:
        return json_query["results"]["bindings"][0][predicate]["value"].replace("http://dbpedia.org/resource/", "").replace("_", " ") + '\n'


def requete_dbpedia_multiple(requete, predicate, entity_of_type="dbo"):
    result = ""
    json_query = json_load(requete, predicate, entity_of_type)

    if(not json_query["results"]["bindings"]):
        return "Aucun résultat correspondant à votre recherche.\n"

    for resultats_requete in json_query["results"]["bindings"]:
        if(not resultats_requete[predicate]["value"].startswith("http://dbpedia.org")):
            result+=resultats_requete[predicate]["value"] + " et "
            continue

        json_result = json.loads(query("""SELECT ?label WHERE {<""" + resultats_requete[predicate]["value"] + """> rdfs:label ?label. FILTER langMatches(lang(?label),"fr")}""", "http://dbpedia.org/sparql"))
        if(json_result["results"]["bindings"]):
            result += json_result["results"]["bindings"][0]["label"]["value"]+ " et "
        else:
            result += resultats_requete[predicate]["value"].replace("http://dbpedia.org/resource/", "").replace("_", " ") + " et "

    return result[:-3]+'\n'
    

if __name__ == '__main__':
    spacy.prefer_gpu()
    nlp = spacy.load("fr_core_news_lg")

    # Configurez True si vous souhaitez afficher des exemples de questions pré-configurés, false sinon
    exemple_questionsxml = True #Exemples tirées du jeu de données fourni: questions.xml
    exemple_autres = True #Autres exemples pré-configurées

    if(exemple_questionsxml):
        question = "Quelle cours d'eau est traversé par le pont de Brooklyn ?"
        print(reponse(question))

        question = "Qui est le créateur de Wikipedia ?"
        print(reponse(question))

        question = "Dans quel pays commence le Nil ?"
        print(reponse(question))

        question = "Dans quel pays se trouve le Nil ?"
        print(reponse(question))

        question = "Quel est l'endroit le plus haut du Karakoram ?"
        print(reponse(question))

        question = "Qui a conçu le pont de Brooklyn ?"
        print(reponse(question))

        question = "Qui est le créateur de Goofy ?"
        print(reponse(question))

        question = "Qui est le maire de New York City ?"
        print(reponse(question))

        question = "Quels sont les pays traversés par l'Ienisseï ?"
        print(reponse(question))

        question = "Dans quel musée est exposé Le Cri de Munch ?"
        print(reponse(question))

        question = "Quels sont les états voisins de l'Illinois ?"
        print(reponse(question))

        question = "Qui était l'épouse du président américain Lincoln ?"
        print(reponse(question))

        question = "En quel langage de programmation a été écrit GIMP ?"
        print(reponse(question))

        question = "Dans quel pays se trouve le lac Limerick ?"
        print(reponse(question))

        question = "Quel est la devise de la Tchéquie ?"
        print(reponse(question))

        question = "Qui a développé World of Warcraft ?"
        print(reponse(question))

        question = "Qui possède Aldi ?"
        print(reponse(question))

        question = "Combien d'employés a IBM ?"
        print(reponse(question))

        question = "Quel est l'indicatif téléphonique de Berlin ?"
        print(reponse(question))

        question = "Quand se déroula la bataille de Gettysburg ?"
        print(reponse(question))

        question = "Quels sont les langues officielles des Philippines ?"
        print(reponse(question))

        question = "Qui a écrit le livre Les Piliers de la terre ?"
        print(reponse(question))

        question = "Quel est la site web de Forbes ?"
        print(reponse(question))

        question = "Quels prix ont été gagnés par Wikileaks ?"
        print(reponse(question))

        question = "Donne-moi tous les acteurs jouant dans le film Last Action Hero."
        print(reponse(question))

        question = "A qui appartient Universal Studios ?"
        print(reponse(question))

        question = "Quel est la cause de décès de Bruce Carver ?"
        print(reponse(question))

    if(exemple_autres):
        question = "Qui est Emmanuel Macron ?"
        print(reponse(question))

        question = "Qui est Nicolas Sarkozy ?"
        print(reponse(question))

        question = "Qui est Zinedine ?"
        print(reponse(question))
    
        question = "Qui est le maire de Paris ?"
        print(reponse(question))

        question = "Qui est le maire de Budapest ?"
        print(reponse(question))

        question = "Qui est le maire de Lyon ?"
        print(reponse(question))

        question = "Qui est le maire de Marseille ?"
        print(reponse(question))

        question = "Qui est le maire de Nice ?"
        print(reponse(question))

        question = "Qui est le président des USA ?"
        print(reponse(question))

        question = "Qui est le président de la France ?"
        print(reponse(question))

        question = "Qui est la chanceliere de l'Allemagne ?"
        print(reponse(question))

        question = "Où est ce que se trouve la ville de Paris ? "
        print(reponse(question))

        question = "Où est ce que se trouve la ville de Tokyo ? "
        print(reponse(question))

        question = "Dans quel pays se trouve la ville de Grenoble ? "
        print(reponse(question))

        question = "Dans quel pays se trouve le lac Limerick ?"
        print(reponse(question))

        question = "Où est Le Caire ? "
        print(reponse(question))
    
        question = "Quelle est la capitale de l'Égypte ?"
        print(reponse(question))
    
        question = "Quelle est la plus grande ville de la France ?"
        print(reponse(question)) 

        question = "Quels sont les états autour du Kansas ?"
        print(reponse(question))

        question = "Quel est la monnaie de la France ?"
        print(reponse(question))

        question = "Qui a écrit le livre Frankenstein ?"
        print(reponse(question))

        question = "Qui a produit le jeu Mario 64 ?"
        print(reponse(question))

    #question = input()
    #print(reponse(question))