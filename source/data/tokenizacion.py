from nltk.tokenize import word_tokenize

def tokenizar(texto):
  texto = " ".join(texto)
  tokenizado = word_tokenize(texto)
  return tokenizado

if __name__ == '__main__':
  tokenizar()