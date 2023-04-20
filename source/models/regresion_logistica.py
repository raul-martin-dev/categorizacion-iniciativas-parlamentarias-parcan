import sys
sys.path.append('..')
sys.path.append('../..')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)


def regresion_logistica(cfg: GlobalConfig):
    df = pd.read_csv(cfg.paths.procesados)
    X = df['asunto']
    y = df['categoria']

    if cfg.mostrado.grafos == True:
      print("\n> Generando gráfico de barras del dataset...")
      count = df['categoria'].value_counts()
      count.plot.bar()
      plt.ylabel('Cantidad de asuntos')
      plt.xlabel('Categoría')
      plt.show()
      print("Vista del gráfico cerrada <\n")

    print("\n-- Método de regresión logística: entrenamiento")
    print("> Comenzando el modelo de precisión...\n")
    
    bow_converter = CountVectorizer(tokenizer=lambda doc: doc)
    X = bow_converter.fit_transform(X)
    
    if cfg.pipeline.balanceo == True:
      print("\n> Empezando el balanceo del dataset...")
      resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
      X, y = resample.fit_resample(X, y)

      if cfg.mostrado.grafos == True:
        count = y.value_counts()
        count.plot.bar()
        plt.ylabel('Cantidad de asuntos')
        plt.xlabel('Categorías')
        plt.show()
        print("Dataset balanceado! <\n")
        
    X_train,X_test, Y_train,Y_test = train_test_split(X,y,test_size=0.2, random_state=25)

    model_precision = LogisticRegression(C=1000, penalty='l2', max_iter=10000, multi_class="ovr").fit(X_train, Y_train)
    pred_tests = model_precision.predict(X_test)
    print("Modelo de precisión completado! <\n")

    print("-- Testeo de la precisión:", precision_score(Y_test, pred_tests, average='weighted'))
    print("-- Testeo del recall:", recall_score(Y_test, pred_tests, average='weighted'))
    print("-- Testeo del f1:", f1_score(Y_test, pred_tests, average='weighted'))
    print("-- Testeo del accuracy:", accuracy_score(Y_test, pred_tests))
    print()

    if cfg.mostrado.matriz:
        # Matriz de confusión
        cm_bow = confusion_matrix(Y_test, pred_tests)

        class_label = Y_test.unique()
        df_cm = pd.DataFrame(cm_bow, index = class_label, columns = class_label)

        print("\n> Generando matriz de confusión...")
        sns.heatmap(df_cm, annot = True, fmt = 'd')
        plt.title('Matriz de confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        plt.show()
        print("Vista de la matriz de confusión cerrada! <\n")

if __name__ == '__main__':
  regresion_logistica()
