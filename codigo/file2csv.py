import csv
import pandas as pd

def guardar_csv(nombre, datos, etiquetaCluster):
    with open(nombre, "w", newline="\n") as f:
        nombre_columna = ['v1', 'v2','v3', 'v4','v5', 'v6','v7', 'v8','v9', 'v10']
        data_writer = csv.DictWriter(f, fieldnames=nombre_columna)
        data_writer.writeheader()
        for dato in datos:
            data_writer.writerows({'v1': dato.fecha, 'v2': dato.temas, 
                                'v3': dato.sitios, 'v4': dato.personas, 
                                'v5': dato.organizaciones, 'v6': dato.intercambios, 
                                'v7': dato.companias, 'v8': etiquetaCluster, 
                                'v9': dato.titulo, 'v10': dato.cuerpo})

def guardar_pandas_csv(nombre, datos, etiquetaCluster):
    
    for dato in datos:
        df = pd.DataFrame([dato.fecha, dato.temas, 
                            dato.sitios, dato.personas, 
                            dato.organizaciones, dato.intercambios, 
                            dato.companias, etiquetaCluster, 
                            dato.titulo, dato.cuerpo])
    
    """
    df = pd.DataFrame([datos[0].fecha, [datos[0].temas], 
                    datos[0].sitios, datos[0].personas, 
                    datos[0].organizaciones, datos[0].intercambios, 
                    datos[0].companias, etiquetaCluster, 
                    datos[0].titulo, datos[0].cuerpo])
    """
    """
    df = pd.DataFrame([datos[0].fecha, datos[0].temas, datos[0].sitios])
    df_tr = df.transpose()
    df_tr.to_csv(nombre, index=False, header=None)
    """
    """
    data ={'Fecha': datos[0].fecha.pop(), 'Temas': datos[0].temas.pop(), 
            'Sitios': datos[0].sitios.pop(), 'Personas': datos[0].personas.pop(), 
            'Orgs': datos[0].organizaciones.pop(), 'Intercambios': datos[0].intercambios.pop(), 
            'Empresas': datos[0].companias.pop(), 'Cluster': etiquetaCluster, 
            'Titulo': datos[0].titulo.pop(), 'Cuerpo': datos[0].cuerpo.pop()}
""""""
    for dato in datos:
        data = {'Fecha': dato.fecha, 'Temas': dato.temas[0]}
        df = pd.DataFrame(data) 
    """
 
    df.to_csv(nombre, index=False)