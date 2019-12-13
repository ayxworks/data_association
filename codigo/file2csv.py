import csv
import pandas as pd

def guardar_csv(nombre, datos, etiquetaCluster, misEtiquetas):
    with open(nombre, "w", newline="\n") as f:
        nombre_columna = ['Fecha', 'Tema1', 'Tema2','Tema3','Sitios', 'Personas','Organizaciones', 'Intercambios','Empresas', 'Etiqueta_cluster', 'Mi_Etiqueta','Titulo', 'Cuerpo']
        data_writer = csv.DictWriter(f, fieldnames=nombre_columna)
        data_writer.writeheader()
        lTemas = []
        lMiCluster = []
  
        for i, dato in enumerate(datos):
            miEtiqueta = ''
            for h in range (3):
                if len(dato.temas)!= 0:
                    lTemas.append(dato.temas.pop())
                else:
                    lTemas.append("nada")
            for j, miEtiqueta in enumerate(misEtiquetas):
                if etiquetaCluster[i] ==  j:
                    miEtiqueta = str("cluster" + str(j) + "_" + misEtiquetas[j])
                    lMiCluster.append(miEtiqueta)
            data_writer.writerow({'Fecha': dato.fecha.pop().split()[0], 'Tema1': lTemas[0], 'Tema2': lTemas[1], 'Tema3': lTemas[2], 
                                'Sitios': dato.sitios.pop(), 'Personas': dato.personas.pop(), 
                                'Organizaciones': dato.organizaciones.pop(), 'Intercambios': dato.intercambios.pop(), 
                                'Empresas': dato.companias.pop(), 'Etiqueta_cluster': etiquetaCluster[i], 'Mi_Etiqueta': lMiCluster[i], 
                                'Titulo': dato.titulo, 'Cuerpo': dato.cuerpo})
