import pandas as pd
def load_emoji_sentiments(file_path):
    # Lee el archivo Excel
    df = pd.read_excel(file_path)

    # Inicializa el diccionario
    emoji_sentiments = {}

    # Itera sobre las filas del DataFrame
    for index, row in df.iterrows():
        # Extrae la información relevante de la fila
        emoji = row['emoji']
        negative = row['negative']
        positive = row['positive']

        # Crea la entrada en el diccionario
        emoji_sentiments[emoji] = {'negative': negative, 'positive': positive}

    return emoji_sentiments
#EMOJIS---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# Ruta al archivo Excel
xlsx_file_path = 'Recursos/Emojis lexicon.xlsx'
corpus = pd.read_excel('Rest_Mex_2022_preprocesado.xlsx')
# Llama a la función para cargar el diccionario
emoji_sentiments_dict = load_emoji_sentiments(xlsx_file_path)

from sklearn.model_selection import train_test_split, KFold, cross_validate
import pandas as pd

import pandas as pd

def get_sentiment_score(text, emoji_sentiments_dict):

  # Elimina los espacios en blanco del texto
  text = text.strip()


  # Divide el texto en palabras
  words = text.split()

  # Inicializa los acumuladores
  positive_score = 0
  negative_score = 0

  # Itera sobre las palabras
  for word in words:
    # Si la palabra es un emoji
    if word in emoji_sentiments_dict:
      # Agrega el puntaje de sentimiento del emoji a los acumuladores
      positive_score += emoji_sentiments_dict[word]['positive']
      negative_score += emoji_sentiments_dict[word]['negative']

  # Regresa el diccionario con los acumuladores
  return {'positive_score': positive_score, 'negative_score': negative_score}

import numpy as np
# Ejemplo de uso
corpus = pd.read_excel('Rest_Mex_2022_preprocesado.xlsx')

emoji_sentiments_dict = load_emoji_sentiments('Recursos/Emojis lexicon.xlsx')

# Combina las columnas 'Title' y 'Opinion' en una nueva columna llamada 'Combined'
corpus['Combined'] = corpus['Title'].astype(str) + ' ' + corpus['Opinion'].astype(str)
# Aplica la función de negación

# Crear conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    corpus['Combined'], corpus['Polarity'], test_size=0.2, random_state=0
)


