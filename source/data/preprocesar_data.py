import sys
sys.path.append('..')
sys.path.append('../..')

from config.config import GlobalConfig
from features.pipeline_preprocesado import preprocesado

from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)

import pandas as pd
import csv

def preprocesar_data(cfg: GlobalConfig):
  df = pd.read_csv(cfg.paths.limpios)
  asuntos = df['asunto'].values.tolist()
  asuntos_preprocesados = []

  print("\n> Running: Pipeline de preprocesado...\n")
  for asunto in asuntos:
    asunto_preprocesado = preprocesado(asunto, lw=cfg.preprocesado.lw, pc=cfg.preprocesado.pc, st=cfg.preprocesado.st, tk=cfg.preprocesado.tk)
    asuntos_preprocesados.append(asunto_preprocesado)
  df['asunto procesado'] = asuntos_preprocesados
  print("Pipeline completado! <\n")

  with open(cfg.paths.procesados, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['asunto', 'categoria']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for index, row in df.iterrows():
        writer.writerow({'asunto': row['asunto procesado'], 'categoria': row['categoria']})
  
if __name__ == '__main__':
  preprocesar_data()