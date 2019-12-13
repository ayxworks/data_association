import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

def reglasApriori(csv_path):
    store_data = pd.read_csv(os.getcwd()+ "/" + csv_path)
    store_data.drop('Titulo', inplace=True, axis=1)
    store_data.drop('Cuerpo', inplace=True, axis=1)
    store_data.drop('Etiqueta_cluster', inplace=True, axis=1)
    store_data.head()
    records = []
    for i, value in enumerate(store_data.values, 0):
        records.append([str(store_data.values[i,j]) for j in range(0, len(value)) if store_data.values[i,j] != "nada"])
    association_rules = apriori(records, min_support=0.001, min_confidence=0.05, min_lift=2, min_length=2)
    association_results = list(association_rules)
    print("Se han generado "+str(len(association_results)) + " reglas")
    return association_results

def print_bonito(reglas):
    for item in reglas:
        # first index of the inner list
        # Contains base item and add item
        print("=====================================")  
        pair = item[0] 
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])

        #second index of the inner list
        print("Support: " + str(item[1]))

        #third index of the list located at 0th
        #of the third index of the inner list

        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")  

def out_txt_reglas(reglas):
    print("Se han generado "+str(len(reglas)) + " reglas")
    with open("Output.txt", "w") as text_file:
        text_file.write("Hay "+str(len(reglas)) + " reglas")
        for item in reglas:
            # first index of the inner list
            # Contains base item and add item
            text_file.write("=====================================\n")  
            pair = item[0] 
            items = [x for x in pair]
            text_file.write("Rule: " + items[0] + " -> " + items[1] +"\n")

            #second index of the inner list
            text_file.write("Support: " + str(item[1])+ "\n")

            #third index of the list located at 0th
            #of the third index of the inner list

            text_file.write("Confidence: " + str(item[2][0][2])+ "\n")
            text_file.write("Lift: " + str(item[2][0][3])+ "\n")
            text_file.write("=====================================\n")  