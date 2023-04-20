import pandas as pd

datos_csv = pd.read_csv('../../data/external/stopwords.csv')
stopwords = datos_csv['stopwords'].values.tolist()

def limpiar_stopwords(texto):
  texto = texto.split()
  texto_limpio = [palabra for palabra in texto if palabra not in stopwords]
  return texto_limpio

if __name__ == '__main__':
  limpiar_stopwords()