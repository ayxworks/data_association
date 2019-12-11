import os
import sys
import time
import preproceso, util, file2csv
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

def runClusteringPruebas(argumentos):
    import __main__
    comienzo = time.time()
    print('\n')
    ##########################################################################################
    if not argumentos.skip_preproceso:
        print('1: Preprocessing')
        tfidf_vecs, documentos = preproceso.preprocesar(argumentos.preproceso)
        print ('Ha tardado en preprocesar ', calc_tiempo(comienzo), 'segundos!')
        
    ##########################################################################################
    if not argumentos.skip_jerarquico:
        print('2: Cluster aglomerativo')
        vector_dataset = util.cargar(os.getcwd() + argumentos.vector_tupla)
        documentos = util.cargar(os.getcwd() + argumentos.backup_datos)
        print('Se ha cargado los vectores tf-idf, del directorio: ' + argumentos.vector_tupla)
        cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        etiquetaCluster = cluster.fit_predict(vector_dataset.toarray())
        print ('Ha tardado en hacer el cluster jerarquico', calc_tiempo(comienzo), ' segundos!')
        #file2csv.guardar_csv(argumentos.path_jerarquico , documentos, etiquetaCluster)
        file2csv.guardar_pandas_csv(argumentos.path_jerarquico , documentos, etiquetaCluster)
        print ('Guardando en csv')
    ##########################################################################################
    if not argumentos.skip_kmeans:
        print('3: Cluster k-means')
        vector_dataset = util.cargar(os.getcwd() + argumentos.vector_tupla)
        docs = util.cargar(os.getcwd() + argumentos.backup_datos)
        kmeans = KMeans(n_clusters=3).fit(vector_dataset)
        etiquetaCluster = kmeans.predict(docs)
        print ('Ha tardado el cluster k-means', calc_tiempo(comienzo), ' segundos!')    
        file2csv.guardar_csv(argumentos.path_kmeans, documentos, etiquetaCluster)
        print ('Guardando en csv')
    
    print ('\nFin del programa: ', calc_tiempo(comienzo), 'segundos!')
    print("Gracias por utilizar nuestro programa\n")

def readCommand( argv ):
    """
    Funcion que permite pasar argumentos en el terminal .
    """
    from optparse import OptionParser
    usageStr = """
    USO:      python main.py <options>
    EJEMPLOS:   (1) python main.py
                    - Se hace el preproceso y el cluster en carpetas predeterminadas
                (2) python main.py --preproceso carpeta_preproceso --clustering carpeta_cluster
                OR  python main.py -l carpeta_preproceso -z carpeta_cluster
                    - Se hace el preproceso y el cluster en carpetas predeterminadas
                    
            Para mas informacion utilizar -h o --help
                python main.py -h
    """
    parser = OptionParser(usageStr)

    parser.add_option('-j','--path_jerarquico', dest='path_jerarquico',
                      help='Coge los temas procesados e imprime por pantalla', default='preproceso/jerarquico.csv')
    parser.add_option('-k','--path_kmeans', dest='path_kmeans',
                      help='Coge la instancia del indice seleccionado e imprime por pantalla', default='preproceso/kmeans.csv')
    parser.add_option('-i', '--indice_instancia', action='store', dest='indice_instancia',
                      help='Se elige el indice de una instancia', default=0)
    parser.add_option('-r','--skip_preproceso', action='store_false', dest='skip_preproceso',
                      help='Flag para saltarse el preproceso', default=True)
    parser.add_option('-s', '--skip_jerarquico', action='store_false', dest='skip_jerarquico',
                      help='Flag para saltarse el clustering', default=True)
    parser.add_option('-t', '--skip_kmeans', action='store_false', dest='skip_kmeans',
                      help='Flag para saltarse la evaluacion', default=True)
    parser.add_option('-u', '--skip_newInst', action='store_false', dest='skip_newInst',
                      help='Flag para saltarse el apartado de anadir nuevas instancias', default=True)
    parser.add_option('-w', '--skip_test', action='store_false', dest='skip_test',
                      help='Flag para saltarse el apartado de anadir instancias del test que no estan en el cluster', default=True)
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
def calc_tiempo(comienzo):
    tiempo = time.time() - comienzo
    tiempo = time.strftime("%H:%M:%S", time.gmtime(tiempo))
    return tiempo

if __name__ == '__main__':
    print('start')
    args = readCommand( sys.argv[1:] )
    runClusteringPruebas(args)
    pass