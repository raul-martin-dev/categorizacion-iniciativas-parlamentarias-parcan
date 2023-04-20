def minusculas(texto):
  texto_minusculas = "".join(palabra.lower() for palabra in texto)
  return texto_minusculas


if __name__ == '__main__':
  minusculas()