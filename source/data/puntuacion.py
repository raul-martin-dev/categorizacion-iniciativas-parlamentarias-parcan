import string

def puntuacion(texto):
  punctuation_words = string.punctuation
  punctuation_words += "¿¡"
  texto_limpio = "".join(palabra for palabra in texto if palabra not in punctuation_words)
  return texto_limpio


if __name__ == '__main__':
  puntuacion()