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

# Ruta al archivo Excel
xlsx_file_path = 'Emojis lexicon.xlsx'

# Llama a la función para cargar el diccionario
emoji_sentiments_dict = load_emoji_sentiments(xlsx_file_path)

# Imprime el diccionario resultante
print(emoji_sentiments_dict)
