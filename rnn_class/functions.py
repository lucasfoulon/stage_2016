def containAtLeastOneWord(text, words):
  for oneWord in words:
    if oneWord in text:
      return True
  return False

def isAtLeastOneWord(char, words):
  for oneWord in words:
    if char == oneWord:
      return True
  return False

def replacePonctuation(term, word):
  if term in word: 
    return list(word.partition(term))
  else:
    return word

def flatten(*args):
    for x in args:
        if hasattr(x, '__iter__'):
            for y in flatten(*x):
                yield y
        else:
            yield x

def find_pui(x):
  n = 0
  while x / 10 >= 1:
    x = x / 10
    n = n + 1
  return n

def dim_table(x):
  pui = find_pui(x)
  if pui == 0: pui += 1
  vart = ( x/( 10**(pui-1) ) )
  if ( x % ( 10**(pui-1) ) ) != 0 :
    vart += 1
  return (10**(pui-1)),vart