"""
Importar modulos y librerias
os: para interactuar con el sistema operativo
nltk: para procesar el texto (linguistica y tokenizacion)
string: para trabajr con el texto y encoding
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os, nltk, string, random, util
from bs4 import BeautifulSoup
from operator import itemgetter
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class ListaDicc:
    def __init__(self):
        self.titulo = []
        self.cuerpo = []

class Datos:
    # Se guardan los temas y sitios, por si se quiere omitir en el clustering o no
    etiquetas_temas_sitios = set()

    def __init__(self, articulo):
        self.titulo = []
        self.cuerpo = []
        self.temas = []
        self.sitios = []
        self.tema_numerico = [] 
        self.fecha = []
        self.personas = []
        self.organizaciones = []
        self.intercambios = []
        self.companias = []
        self.palabras = ListaDicc()
        self.asignarTemaArticulo(articulo)
        self.asignarEtiquetasVarias(articulo)

    ### funciones ###
    def asignarTemaArticulo(self, articulo):
        """ funcion que asugna la etiqueta tema del articulo """
        for temas in articulo.find_all("topics"):
            if len(temas)==0:
                self.temas.append("nada") 
            else:
                for tema in temas:
                    un_tema = tema.get_text()
                    #un_tema = tema.text.encode('utf-8', 'ignore')
                    self.temas.append(un_tema)
                    Datos.etiquetas_temas_sitios.add(un_tema)

    def asignarTemaNumerico(self, lista_temas):
        """ funcion que asugna la etiqueta tema numerico para evaluacion del cluster """
        for tema in self.temas:
            try:
                self.tema_numerico.append(lista_temas.index(tema))        
            except ValueError:
                print(tema + ' ,no esta en la lista y se ha añadido')
                lista_temas.append(tema)
                self.tema_numerico.append(lista_temas.index(tema))

    def asignarEtiquetasVarias(self, articulo):
        """ funcion que asugna la etiqueta tema del articulo """
        for dates in articulo.find_all("date"):
            if len(dates)==0:
                self.fecha.append("nada") 
            else:
                for fecha in dates:
                    una_fecha = str(fecha)
                    self.fecha.append(una_fecha)
        
        for personas in articulo.find_all("people"):
            if len(personas)==0:
                self.personas.append("nada") 
            else:
                for persona in personas:
                    una_persona = persona.get_text()
                    self.personas.append(una_persona)
        
        for sitios in articulo.find_all("places"):
            if len(sitios)==0:
                self.sitios.append("nada") 
            else:
                un_lugar = sitios.text.encode('utf-8', 'ignore')
                self.sitios.append(str(un_lugar.decode("utf-8")))
                Datos.etiquetas_temas_sitios.add(un_lugar)
        
        for organizaciones in articulo.find_all("ORGS"):
            if len(organizaciones)==0:
                self.organizaciones.append("nada") 
            else:
                for org in organizaciones:
                    una_org = org.get_text()
                    self.organizaciones.append(una_org)
        
        for cambios in articulo.find_all("EXCHANGES"):
            if len(cambios)==0:
                self.intercambios.append("nada") 
            else:
                for cambio in cambios:
                    un_cambio = cambio.get_text()
                    self.intercambios.append(un_cambio)
        
        for empresas in articulo.find_all("COMPANIES"):
            if len(empresas)==0:
                self.companias.append("nada") 
            else:
                for empresa in empresas:
                    una_empresa = empresa.get_text()
                    self.companias.append(una_empresa)
    

    def aumentar_lista_dicc(self, articulo):
        """ crea una lista de tokens de las palabras del titulo/cuerpo """
        texto = articulo.find('text')
        titulo = texto.title
        cuerpo = texto.body
        boolEtiqueta = False
        if titulo != None:
            self.palabras.titulo = self.tokenizacion(titulo.text, boolEtiqueta)
            self.titulo = titulo.text
        if cuerpo != None:
            self.palabras.cuerpo = self.tokenizacion(cuerpo.text, boolEtiqueta)
            self.cuerpo = cuerpo.text

    def tokenizacion(self, texto, bool_etiquetas = False):
        """
            Dado un texto parseado por beautifulSoup se anaden a la clase los tokens del articulo 
            Pre: El texto a analizar
            Post: Se crean los tokens para luego analizar y darles peso 
        """
        """ crea la lista de palabras que analizaremos  """
        # quita digitos y puntuacion
        sin_num = texto.translate(string.digits)
        sin_punt = sin_num.translate(string.punctuation)
        """pruebas no funciona con lo de abajo"""
        #text = texto.encode("utf8").translate(None, string.digits).decode("utf8")
        #sin_puntuacion = text.encode("utf8").translate(None, string.punctuation).decode("utf8")

        # separar el texto en tokens
        tokens = nltk.word_tokenize(sin_punt)
        # quitar si el usuario quiere la 'clase'/etiquetas/temas, stop words y palabras que no sean en ingles
        sin_stop = [w for w in tokens if not w in stopwords.words('english')]
        if bool_etiquetas:
            clase_etiqueta_art = [w for w in sin_stop]
        else:
            clase_etiqueta_art = [w for w in sin_stop if not w in Datos.etiquetas_temas_sitios]

        eng = [y for y in clase_etiqueta_art if wordnet.synsets(y)]

        # lemmatizacion
        lemmas = []
        lematizador = WordNetLemmatizer()
        for token in eng:
            lemmas.append(lematizador.lemmatize(token))
        # sacar raices
        raices = []
        saca_raices = PorterStemmer()
        for token in lemmas:
            raiz = saca_raices.stem(token).encode('utf-8', 'ignore')
            if len(raiz) >= 4:
                raices.append(raiz.decode('utf-8', 'ignore'))
        return raices


fichero_datos_vectores = ['datasets/dataset1.csv', 'datasets/dataset2.csv']

class Tf_Idf:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_df=0.9)
        self.palabras = []
        self.vPesos = []

    def generar_vocab_npalabras(self, docs):
        """
        Dados los documentos y un path se genera los tf-idf de todas las palabras que se incluyen en el articulo, titulo y cuerpo
        Pre: Los documentos y el path para guardar todos los tokens totales (no se van a usar todos)
        Post: Se guardan todos los tokens y se devuelve los pesos de cada token de cada instancia
        """
        palabras_dicc = dict([])
        for i, doc in enumerate(docs):
            palabras_dicc[i] = ' '.join(doc.palabras.titulo + doc.palabras.cuerpo)
        del(docs)
        self.vPesos = self.tfidf.fit_transform(palabras_dicc.values())
        self.palabras = self.tfidf.get_feature_names()
        print("Espacio vectorial analizado y valores tf_idf calculados")

#######################################################################
#            funciones del preproceso, cargar los archivos            #
#######################################################################
def scrap_texto(texto_plano):
    return BeautifulSoup(texto_plano, "html.parser")

def escanear_docs(directorio):
    """
        Dado un directorio se leen todos los archivos y se filtran los articulos
        Pre: El directorio a examinar
        Post: Los documentos parseados listos para python
    """
    pares = dict([])
    documentos = []
    for fichero in os.listdir(directorio):
        # abrir los archivos 'xxx.sgm' de un directorio
        docs = open(os.path.join(directorio, fichero), 'r')
        texto = docs.read()
        docs.close()
        bsoup = scrap_texto(texto.lower())
        for reuter in bsoup.find_all("reuters"):
            articulo = Datos(reuter)
            pares[articulo] = reuter

        for articulo, reuter in pares.items():
            articulo.aumentar_lista_dicc(reuter)
            documentos.append(articulo)  
        print("Se ha terminado de examinar el fichero:", fichero)
    return documentos

def crearListaTemasTotales(documentos):
    """
        Dadas las instancias genera una lista de todos los temas
        Pre: Los documentos a analizar
        Post: Se devuelve las etiquetas tema
    """
    lista = set()
    lista.add("nada")
    for doc in documentos:
        for tema in doc.temas:
            lista.add(tema)
    #print(lista)
    return lista

def shuffle_split(directorio):
    """
        Dado un directorio escanea los doccumentos, los barajea y los separa en 80-20
        Pre: El directorio a escanear
        Post: Los datos para entrenar y probar
    """
    documentos = escanear_docs(directorio)
    random.shuffle(documentos)
    train_data = documentos[:int((len(documentos)+1)*.80)]
    test_data = documentos[int(len(documentos)*.80+1):]
    return train_data, test_data

def preprocesar(directorio_ruta):
    """
        Dada una ruta de datos se genera el dataset de las instancias nuevas
        Pre: El directorio de los datos
        Post: La lista de instancias con tuplas de pesos tf-idf de cada instancia
    """
    #directorio = 'datos'
    print('\nGenerando los vectores de las instancias')
    documentos = escanear_docs(directorio_ruta)
    random.shuffle(documentos)
    tfidf = Tf_Idf()
    tfidf.generar_vocab_npalabras(documentos)
    util.guardar(os.getcwd()+"/preproceso/full_lista_articulos.txt", documentos)
    util.guardar(os.getcwd()+"/preproceso/full_tfidf.txt", tfidf.vPesos)
    util.guardar(os.getcwd()+"/preproceso/raw_full_tfidf", tfidf)
    print('Preproceso completado!')
    return tfidf.vPesos, documentos

def preprocesar_train(directorio_ruta):
    """
        Dada una ruta de datos se genera el dataset de las instancias nuevas
        Pre: El directorio de los datos
        Post: La lista de instancias con tuplas de pesos tf-idf de cada instancia
    """
    #directorio = 'datos'
    print('\nGenerando los vectores de las instancias')
    train, test = shuffle_split(directorio_ruta)
    lista = list(crearListaTemasTotales(train) | crearListaTemasTotales(test))
    print("Hay " + str(len(lista)) + " temas totales en el conjunto de datos analizados")
    for doc in train:
        doc.asignarTemaNumerico(lista)
    util.guardar(os.getcwd()+"/preproceso/lista_temas.txt", lista)
    tfidf = Tf_Idf()
    tfidf.generar_vocab_npalabras(train)
    util.guardar(os.getcwd()+"/preproceso/lista_articulos_train.txt", train)
    util.guardar(os.getcwd()+"/preproceso/lista_articulos_test.txt", test)
    util.guardar(os.getcwd()+"/preproceso/train_tfidf.txt", tfidf.vPesos)
    util.guardar(os.getcwd()+"/preproceso/raw_tfidf", tfidf)
    print('Preproceso completado!')
    return tfidf.vPesos, train

def preprocesar_test(tfidf_path, train_path, test_path, vocabulario_path, lista_temas_path):
    """
        Dado el dataset y las nuevas instancias se obtiene un nuevo dataset y sus pesos correspondientes en tf-idf
        Pre: Los pesos tf-idf anteriores, el path de los datos anteriores, los datos nuevos, el path del nuevo dataset y el paz de todos los temas
        Post: Se genera otro dataset tf-idf con las nuevas instancias
    """
    print("generar test tfidf")
    test = util.cargar(os.getcwd()+ test_path)
    tfidf = util.cargar(os.getcwd()+ tfidf_path)
    print('\nGenerando los vectores de las instancias')
    tfidf.generar_vocab_npalabras(test)
    util.guardar(os.getcwd()+"/preproceso/test_tfidf.txt", tfidf.vPesos)
    util.guardar(os.getcwd()+"/preproceso/raw_tfidf_test", tfidf)
    print('Preproceso completado!')
    return tfidf.vPesos, tfidf

def instancia_articulo(indice, documentos):
    """
        Dada una instancia devuelve el articulo
        Pre: El indice en el cluster y los documentos con los que se ha formado el cluster
        Post: Se imprime por pantalla el articulo deseado
    """
    print("Titulo:")
    print(documentos[indice].titulo)
    print("Cuerpo:")
    print(documentos[indice].cuerpo)
    print("Temas:")
    print(documentos[indice].temas)

def temas_totales_print(temas):
    print(temas)


