# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:00:38 2019

@author: StromValhalla
"""
import pickle
from operator import itemgetter
from collections import OrderedDict

"""
Guarda la estructura de datos
Pre : El path debe existir
Post: El archivo con la estructura de datos guardada
"""
def guardar(path, archivo):
    with open(path, "wb") as res:
        pickle.dump(archivo, res)     
        
    res.close()


"""
Carga la estructura de datos
Pre : El path debe existir
Post: La estructura de datos
"""
def cargar(path):
    with open(path, "rb") as fp:  
        obj = pickle.load(fp)
        
    fp.close()

    return obj

def etiquetaClusterTema(datos, etiquetaCluster):
    clusterTemas = []
    temaEscogido = []
    misDict = []
    
    #crear una lista por cada cluster que exista
    for i in range(len(set(etiquetaCluster))):
        aux = []
        clusterTemas.append(aux)
    #anadir cada tema a la lista del cluster que sea
    for j in range(len(datos)):
        if datos[j].temas != 0:
            clusterTemas[etiquetaCluster[j]].append(datos[j].temas)
    #contar los temas en cada lista
    for listaInst in clusterTemas:
        my_dict = {}
        for listaTemas in listaInst:
            for tema in listaTemas:
                if tema != "nada":
                    if tema in my_dict:
                        my_dict[tema] += 1
                    else:
                        my_dict[tema] = 1
        misDict.append(my_dict)
    
    for k in range(len(clusterTemas)):
        d = OrderedDict(sorted(misDict[k].items(), key=itemgetter(1)))
        keys = list(d.keys())
        if len(keys) > 1:
            etiqueta = keys[0] + "_" + keys[1]
        else:
            etiqueta = keys[0]
        temaEscogido.append(etiqueta)
    return temaEscogido