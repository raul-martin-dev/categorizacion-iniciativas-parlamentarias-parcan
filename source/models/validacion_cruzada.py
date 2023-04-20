import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score

import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedKFold
from statistics import mean

def validacion_cruzada(cfg: GlobalConfig):
  print("\n-- Modelo de regresión logística: entrenamiento con validación cruzada")
  df = pd.read_csv(cfg.paths.procesados)
  X = df['asunto']
  y = df['categoria']

  model = LogisticRegression(C=1000, penalty='l2', max_iter=10000, multi_class='ovr')
  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
  acc_stratified = []
  bow_converter = CountVectorizer(tokenizer=lambda doc: doc)
  num = 1
  
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

  for train_index, test_index in skf.split(X, y):
    print(f"> Comenzando modelo de precisión {num}...")
    X_train_fold, X_test_fold = X[train_index], X[test_index] 
    y_train_fold, y_test_fold = y[train_index], y[test_index]


    model.fit(X_train_fold, y_train_fold) 
    acc_stratified.append(model.score(X_test_fold, y_test_fold))
    
    print("> Testeando modelo...\n")
    print(f"-- Test {num}: \n")
    pred_tests = model.predict(X_test_fold)

    print("-- Test precision:", precision_score(y_test_fold, pred_tests, average='weighted'))
    print("-- Test recall:", recall_score(y_test_fold, pred_tests, average='weighted'))
    print("-- Test f1:", f1_score(y_test_fold, pred_tests, average='weighted'))
    print("-- Test accuracy:", accuracy_score(y_test_fold, pred_tests))
    print()
    print(f"\nModelo de precisión {num} e informe completado! <\n")
    num += 1

  print("> Final report:\n")
  print('-- Maximum Accuracy',max(acc_stratified)) 
  print('-- Minimum Accuracy:',min(acc_stratified)) 
  print('-- Overall Accuracy:',mean(acc_stratified))
  print()

if __name__ == '__main__':
  validacion_cruzada()