import os
import sys
import time
import datetime
import preproceso, util, file2csv, asociacion
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

def runClusteringPruebas(argumentos):
    import __main__
    comienzo = time.time()
    print('\n')
    ##########################################################################################
    if not argumentos.all:
        print('-----------Ejecucion completa-----------')
        argumentos.skip_preproceso = argumentos.skip_jerarquico = argumentos.skip_kmeans = argumentos.skip_reglas = False
    ##########################################################################################
    if not argumentos.skip_preproceso:
        print('1: Preprocessing')
        tfidf_vecs, documentos = preproceso.preprocesar(argumentos.preproceso)
        print ('Ha tardado en preprocesar ', calc_tiempo_trans(comienzo), 'segundos!')
        
    ##########################################################################################
    if not argumentos.skip_jerarquico:
        print('2: Cluster aglomerativo')
        vector_dataset = util.cargar(os.getcwd() + argumentos.vector_tupla)
        documentos = util.cargar(os.getcwd() + argumentos.backup_datos)
        print('Se ha cargado los vectores tf-idf, del directorio: ' + str(len(documentos)))
        cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        miCluster = cluster.fit_predict(vector_dataset.toarray())
        etiquetas = cluster.labels_
        print ('Ha tardado en hacer el cluster jerarquico', calc_tiempo_trans(comienzo), ' segundos!')
        miEtiquetaElegida = util.etiquetaClusterTema(documentos, etiquetas)
        print ('Ha tardado en escoger un label para el cluster ', calc_tiempo_trans(comienzo), ' segundos!')
        file2csv.guardar_csv(argumentos.path_jerarquico , documentos, etiquetas, miEtiquetaElegida)
        print ('Ha tardado en guardar en csv ', calc_tiempo_trans(comienzo) , ' segundos!')
        plt.figure(figsize=(10, 7))
        plt.scatter(vector_dataset.toarray()[:,0], vector_dataset.toarray()[:,1], c=cluster.labels_, cmap='rainbow')
        plt.show()
        print ('Guardando en csv')
    ##########################################################################################
    if not argumentos.skip_kmeans:
        print('3: Cluster k-means')
        vector_dataset = util.cargar(os.getcwd() + argumentos.vector_tupla)
        docs = util.cargar(os.getcwd() + argumentos.backup_datos)
        print("Se han cargado: " + str(len(docs)) + " instancias")
        miCluster = KMeans(n_clusters=3)
        miCluster.fit_predict(vector_dataset)
        etiquetas = miCluster.labels_
        print ('Ha tardado en hacer el cluster k-means ', calc_tiempo_trans(comienzo), ' segundos!')
        miEtiquetaElegida = util.etiquetaClusterTema(docs, etiquetas)
        print ('Ha tardado en escoger un label para el cluster ', calc_tiempo_trans(comienzo), ' segundos!')
        file2csv.guardar_csv(argumentos.path_kmeans, docs, etiquetas, miEtiquetaElegida)
        print ('Ha tardado en guardar en csv ', calc_tiempo_trans(comienzo) , ' segundos!')
        plt.figure(figsize=(10, 7))
        plt.scatter(vector_dataset.toarray()[:,0], vector_dataset.toarray()[:,1], c=miCluster.labels_, cmap='rainbow')
        plt.show()
        print ('Ha tardado el cluster k-means', calc_tiempo_trans(comienzo), ' segundos!')    

        print ('Guardando en csv')
    ##########################################################################################
    if not argumentos.skip_reglas:
        print('4: Reglas')
        vector_dataset = util.cargar(os.getcwd() + argumentos.vector_tupla)
        docs = util.cargar(os.getcwd() + argumentos.backup_datos)
        res = asociacion.reglasApriori(argumentos.path_jerarquico)
        util.guardar(os.getcwd() + "reglas", res)
        asociacion.print_bonito(res)
        asociacion.out_txt_reglas(res)
    ##########################################################################################
    if not argumentos.all:
        if sorted(cluster.labels_) == sorted(miCluster.labels_):
            print("El cluster jerarquico y k-means han sido iguales")

    print ('\nFin del programa: ', calc_tiempo_trans(comienzo), 'segundos!')
    print("Gracias por utilizar nuestro programa\n")

def readCommand( argv ):
    """
    Funcion que permite pasar argumentos en el terminal .
    """
    from optparse import OptionParser
    usageStr = """
    USO:      python main.py <options>
    EJEMPLOS:   (1) python main.py
                    - Se hace el preproceso y el clustering jerarquico y kmeans; Y se sacan unas reglas de asociacion
                (2) python main.py --skip_preproceso --skip_kmeans
                OR  python main.py -r -t
                    - Se hace el preproceso y el cluster k means en carpetas predeterminadas
                    
            Para mas informacion utilizar -h o --help
                python main.py -h
    """
    parser = OptionParser(usageStr)

    parser.add_option('-j','--path_jerarquico', dest='path_jerarquico',
                      help='Opcion para cambiar la ruta del csv al hacer el cluster jerarquico', default='preproceso/jerarquico.csv')
    parser.add_option('-k','--path_kmeans', dest='path_kmeans',
                      help='Coge la instancia del indice seleccionado e imprime por pantalla', default='preproceso/kmeans.csv')
    parser.add_option('-r','--skip_preproceso', action='store_false', dest='skip_preproceso',
                      help='Flag para realizar el preproceso', default=True)
    parser.add_option('-s', '--skip_jerarquico', action='store_false', dest='skip_jerarquico',
                      help='Flag para realizar la clusterzacion jerarquica', default=True)
    parser.add_option('-t', '--skip_kmeans', action='store_false', dest='skip_kmeans',
                      help='Flag para realizar la clusterizacion kmeans', default=True)
    parser.add_option('-a', '--all', action='store_false', dest='all',
                      help='Flag realizar todo', default=True)
    parser.add_option('-u', '--skip_reglas', action='store_false', dest='skip_reglas',
                      help='Flag realizar la asociacion de reglas', default=True)
    parser.add_option('-b', '--backup_datos', dest='backup_datos',
                      help='Path del archivo donde se guardan las instancias', default='/preproceso/full_lista_articulos.txt')
    parser.add_option('-v', '--backup_datos_test', dest='backup_datos_test',
                      help='Path del archivo donde se guardan las instancias para test', default='/preproceso/lista_articulos_test.txt')
    parser.add_option('-p', '--preproceso', dest='preproceso',
                      help='Path de los textos', default='datos')
    parser.add_option('-c', '--vector_tupla', action='store', dest='vector_tupla',
                      help='Path de los vectores para hacer el cluster', default='/preproceso/full_tfidf.txt')
    parser.add_option('-n', '--newInst', action='store', dest='newInst',
                      help='Para añadir nuevas instancias al cluster', default='test')
    parser.add_option('-d', '--distancia', action='store', dest='distancia',
                      help='Elegir la ecuación para las distancias 1=manhattan, 2=euclidea (predeterminado)', type="int", default=2)

    options, otros = parser.parse_args(argv)
    if len(otros) != 0:
        raise Exception('No se ha entendido este comando: ' + str(otros))
    return options
def calc_tiempo_trans(comienzo):
    tiempo = time.time() - comienzo
    tiempo = time.strftime("%H:%M:%S", time.gmtime(tiempo))
    return tiempo

if __name__ == '__main__':
    print('start')
    args = readCommand( sys.argv[1:] )
    runClusteringPruebas(args)
    pass