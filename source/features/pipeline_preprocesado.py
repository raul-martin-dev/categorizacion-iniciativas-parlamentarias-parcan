import sys
sys.path.append('..')

import pandas as pd
from data.minusculas import minusculas
from data.puntuacion import puntuacion
from data.stopword import limpiar_stopwords
from data.tokenizacion import tokenizar

def preprocesado(text, lw, pc, st, tk):
  print('> Empezando el preprocesado...')
  print('-- text raw: ', text)
  if lw == 1:
    text = minusculas(text)
    print('> Minúsculas: Hecho')
  if pc == 1:
    text = puntuacion(text)
    print('> Puntuación: Hecho')
  if st == 1:
    text = limpiar_stopwords(text)
    print('> Stopwords: Hecho')
  if tk == 1:
    text = tokenizar(text)
    print('> Tokenizado: Hecho')
  print('Preprocesado completado! <\n')
  return text

if __name__ == '__main__':
    preprocesado()

