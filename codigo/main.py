import os
import sys
import time
import preproceso
import util, clustering, evaluador, explorar

def runClusteringPruebas(argumentos):
    import __main__
    comienzo = time.time()
    print('\n')
    ##########################################################################################
    if not argumentos.skip_preproceso:
        print('1: Preprocessing')
        #directorio_ruta = 'datos'
        print(argumentos)
        tfidf_vecs, documentos = preproceso.preprocesar_train(argumentos.preproceso)

        print ('Ha tardado en preprocesar ', calc_tiempo(comienzo), 'segundos!')
    ##########################################################################################
    if not argumentos.skip_clustering:
        print('2: Clustering')
        vector_dataset = util.cargar(os.getcwd() + argumentos.vector_tupla)
        print('Se ha cargado los vectores tf-idf, del directorio: ' + argumentos.vector_tupla)
        cl = clustering.Cluster(vector_dataset)
        cl.clustering(argumentos.distancia)
        print ('Ha tardado en hacer el cluster jerarquico ', calc_tiempo(comienzo), 'segundos!')
    ##########################################################################################
    if not argumentos.skip_evaluacion:
        print('3: Evaluando')
        ev = evaluador.Evaluador()
        path = util.cargar(os.getcwd()+ argumentos.evaluacion)
        instancias =  util.cargar(os.getcwd()+argumentos.vector_tupla)
        ev.evaluar(path, instancias)
        print ('Ha tardado en evaluar ', calc_tiempo(comienzo), 'segundos!')    
    
    ##########################################################################################
    if not argumentos.skip_newInst:
        print('4: Anadir nuevas instancias')
        vectoresTest, ndocs, nNew = preproceso.preprocesar_newInst('/preproceso/raw_tfidf', argumentos.backup_datos, argumentos.newInst, "/preproceso/vocabulario_train.txt", "/preproceso/lista_temas.txt")
        instancias = util.cargar(os.getcwd()+argumentos.vector_tupla)
        instsAClasif = list(range(ndocs, ndocs+nNew))
        datosTest=util.cargar(os.getcwd()+'/preproceso/new_lista_articulos.txt')
        lista_temas = util.cargar(os.getcwd()+'/preproceso/new_lista_temas.txt')
        #instsAClasif = list(range(1, nNew+1))
        agrupacion = explorar.agruparInstanciasPorCluster('/resultados/iter.txt',instancias,3,instsAClasif, vectoresTest, datosTest)
        for each in agrupacion:
            print ("Instancia: " + str(each[0] +1) +", tema estimado: " +  str(each[2]) + ", Tema real: "+ str(each[3]))
            if len(each[3])>0:
                for temaN in each[3]:
                    print("El tema "+ str(temaN) + " es " + str(lista_temas[temaN]))  
        print ('Ha tardado en anadir una nueva instancia ', calc_tiempo(comienzo), 'segundos!')
    ##########################################################################################
    #no funciona de momento
    if not argumentos.skip_test:
        print('5: Anadir nuevas instancias del conjunto separado test')
        vector_dataset_test, ndocs, nNew = preproceso.preprocesar_test('/preproceso/raw_tfidf', argumentos.backup_datos, argumentos.backup_datos_test, "/preproceso/vocabulario_train.txt", "/preproceso/lista_temas.txt")
        instancias = util.cargar(os.getcwd()+argumentos.vector_tupla)
        vectoresTest = util.cargar(os.getcwd()+'/preproceso/test_tfidf.txt')
        datosTest=util.cargar(os.getcwd()+argumentos.backup_datos_test)
        lista_temas = util.cargar(os.getcwd()+'/preproceso/new_lista_temas.txt')
        instsAClasif = list(range(0, nNew+1))
        agrupacion = explorar.agruparInstanciasPorCluster('/resultados/iter.txt',instancias,3,instsAClasif, vector_dataset_test, datosTest)
        for each in agrupacion:
            print (each[2],each[3])
            print(type(each[3]))
            if len(each[3])>0:
                for temaN in each[3]:
                    print("El tema "+ str(temaN) + " es " + str(lista_temas[temaN]))  
        
        print ('Ha tardado en anadir una nueva instancia ', calc_tiempo(comienzo), 'segundos!')
    ##########################################################################################
    if not argumentos.get_instance:
        documentos = util.cargar(os.getcwd()+argumentos.backup_datos)
        preproceso.instancia_articulo(argumentos.indice_instancia, documentos)
        print ('Se ha tardado en buscar la instancia ', calc_tiempo(comienzo), 'segundos!')
    ##########################################################################################
    if not argumentos.get_temas:
        temas = util.cargar(os.getcwd()+"/preproceso/lista_temas.txt")
        preproceso.temas_totales_print(temas)
        print ('Se ha tardado en buscar la instancia ', calc_tiempo(comienzo), 'segundos!')    
    
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
    parser.add_option('-z', '--testing', action='store', dest='testing',
                      help='Pruebas', default='lista_articulos_test')
    parser.add_option('-g','--get_instance', action='store_false', dest='get_instance',
                      help='Coge la instancia del indice seleccionado e imprime por pantalla', default=True)
    parser.add_option('-j','--get_temas', action='store_false', dest='get_temas',
                      help='Coge los temas procesados e imprime por pantalla', default=True)
    parser.add_option('-i', '--indice_instancia', action='store', dest='indice_instancia',
                      help='Se elige el indice de una instancia', default=0)
    parser.add_option('-r','--skip_preproceso', action='store_false', dest='skip_preproceso',
                      help='Flag para saltarse el preproceso', default=True)
    parser.add_option('-s', '--skip_clustering', action='store_false', dest='skip_clustering',
                      help='Flag para saltarse el clustering', default=True)
    parser.add_option('-t', '--skip_evaluacion', action='store_false', dest='skip_evaluacion',
                      help='Flag para saltarse la evaluacion', default=True)
    parser.add_option('-u', '--skip_newInst', action='store_false', dest='skip_newInst',
                      help='Flag para saltarse el apartado de anadir nuevas instancias', default=True)
    parser.add_option('-w', '--skip_test', action='store_false', dest='skip_test',
                      help='Flag para saltarse el apartado de anadir instancias del test que no estan en el cluster', default=True)
    parser.add_option('-a', '--asignar_cluster', dest='asignar_cluster',
                      help='Elegir un cluster si ya hay una estructura', default='/resultados/datosAL.txt')
    parser.add_option('-e', '--evaluacion', dest='evaluacion',
                      help='Se hace la evaluacion del cluster', default='/resultados/dist.txt')
    parser.add_option('-y', '--iteraciones', dest='iteraciones',
                      help='Path de las iteraciones del cluster', default='/resultados/iteraciones.txt')
    parser.add_option('-b', '--backup_datos', dest='backup_datos',
                      help='Path del archivo donde se guardan las instancias', default='/preproceso/lista_articulos_train.txt')
    parser.add_option('-v', '--backup_datos_test', dest='backup_datos_test',
                      help='Path del archivo donde se guardan las instancias para test', default='/preproceso/lista_articulos_test.txt')
    parser.add_option('-p', '--preproceso', dest='preproceso',
                      help='Path de los textos', default='datos')
    parser.add_option('-c', '--vector_tupla', action='store', dest='vector_tupla',
                      help='Path de los vectores para hacer el cluster', default='/preproceso/train_tfidf.txt')
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