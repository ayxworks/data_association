# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:00:38 2019

@author: StromValhalla
"""
import pickle

"""
    Dado una lista de vectores te devuelve su centroide
    pre: Una lista no vacia de tuplas numericas
    post: Devuelve el centroide de esos vectores
"""
def calcularCentro(lista, vectores):
    centro = []
    i=0
    x=0
    
    while i<len(vectores[0]):
        for each in lista: 
            vector = vectores[each]
            x+=vector[i]
            
        x=x/len(lista)
        centro.append(x)
        x=0
        i+=1
    
    centro = tuple(centro)
    return centro


"""
    Se encarga de actualizar las lista con los clusters que se han asignado a cada instancia
    pre: una lista de vectores, las agrupaciones y una lista
    post: Devuelve la lista con la correspondiente agrupacion
"""
def listaClusters(inst, agrup, lista): 
    i = 0
    while i<len(agrup):
        for each in agrup[i]:
            "indice = inst.index(each)"
            lista[each] = i
            
        i+=1
        
    return lista



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
        clust = pickle.load(fp)
        
    fp.close()

    return clust


"""
Genera una lista con ceros
Pre : La longitud de la lista
Post: U lista
"""
def generarLista(num):
    l = [0] * num
    return l


"""
Calcula la distancia manhattan entre dos centroides.
Pre : Coordenadas de dos centroides y valor m
     m=1 -> Distancia Manhattan
     m=2 -> Distancia Euclidea
     m=7.5 -> Distancia Minkowski
Post: Distancia Manhattan, Euclidea o Minkowski entre los dos centroides.
"""
def calcularDistancia(centr1, centr2, m):
    dist=0
    i=0
    while i<len(centr1):
        dist+= (abs(centr1[i]-centr2[i]))**m
        i+=1
    dist = dist**(1/m)
    return dist