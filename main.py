"""
In this practice, the polarity of users' opinions on tourist sites will be determined
The reviews are about restaurants, hotels and tourist attractions
Each opinion is rated with a value from 1 to 5, where:
1 is very negative
2 is negative
3 is neutral
4 is positive
5 is very positive
The aim of the practice is to improve the performance obtained by the basic shared model

Specifications

Corpus preprocessing
Text representation
Train Machine Learning model
Predict Sentiment Polarity

Load the Rest_Mex_2022 corpus
Apply normalization process
Create a training set with 80% of the data and a test set with the remaining 20% of the data (set random_state = 0)

Create different text representation of the corpus

Split the training set into 5 folds. You can use the KFold or cross_validate functions
Train different Machine Learning models tuning parameters (when required) using the 5 folds and calculate the average f1 macro
Select the best adjusted model
Train the selected model using the full train set


Use the trained model to predict the opinions of the test set
Calculate the average f1 macro



"""
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import re
import numpy as np
from scipy.sparse import hstack
from textblob import TextBlob
from spellchecker import SpellChecker
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
# Load the Rest_Mex_2022 corpus (replace 'your_data.csv' with the actual file name)


# Apply normalization process (example: lowercase and remove punctuation)


import pandas as pd




def load_sel():
	#~ global lexicon_sel
	lexicon_sel = {}
	input_file = open('Recursos\SEL_full.txt', 'r')
	for line in input_file:
		#Las líneas del lexicon tienen el siguiente formato:
		#abundancia	0	0	50	50	0.83	Alegría
		
		palabras = line.split("\t")
		palabras[6]= re.sub('\n', '', palabras[6])
		pair = (palabras[6], palabras[5])
		if lexicon_sel:
			if palabras[0] not in lexicon_sel:
				lista = [pair]
				lexicon_sel[palabras[0]] = lista
			else:
				lexicon_sel[palabras[0]].append (pair)
		else:
			lista = [pair]
			lexicon_sel[palabras[0]] = lista
	input_file.close()
	del lexicon_sel['Palabra']; #Esta llave se inserta porque es parte del encabezado del diccionario, por lo que se requiere eliminar
	#Estructura resultante
		#'hastiar': [('Enojo\n', '0.629'), ('Repulsi\xf3n\n', '0.596')]
	return lexicon_sel

def getSELFeatures(cadenas, lexicon_sel):
	#'hastiar': [('Enojo\n', '0.629'), ('Repulsi\xf3n\n', '0.596')]
	features = []
	for cadena in cadenas:
		valor_alegria = 0.0
		valor_enojo = 0.0
		valor_miedo = 0.0
		valor_repulsion = 0.0
		valor_sorpresa = 0.0
		valor_tristeza = 0.0
		cadena_palabras = re.split('\s+', cadena)
		dic = {}
		for palabra in cadena_palabras:
			if palabra in lexicon_sel:
				caracteristicas = lexicon_sel[palabra]
				for emocion, valor in caracteristicas:
					if emocion == 'Alegría':
						valor_alegria = valor_alegria + float(valor)
					elif emocion == 'Tristeza':
						valor_tristeza = valor_tristeza + float(valor)
					elif emocion == 'Enojo':
						valor_enojo = valor_enojo + float(valor)
					elif emocion == 'Repulsión':
						valor_repulsion = valor_repulsion + float(valor)
					elif emocion == 'Miedo':
						valor_miedo = valor_miedo + float(valor)
					elif emocion == 'Sorpresa':
						valor_sorpresa = valor_sorpresa + float(valor)
		dic['__alegria__'] = valor_alegria
		dic['__tristeza__'] = valor_tristeza
		dic['__enojo__'] = valor_enojo
		dic['__repulsion__'] = valor_repulsion
		dic['__miedo__'] = valor_miedo
		dic['__sorpresa__'] = valor_sorpresa
		
		#Esto es para los valores acumulados del mapeo a positivo (alegría + sorpresa) y negativo (enojo + miedo + repulsión + tristeza)
		dic['acumuladopositivo'] = dic['__alegria__'] + dic['__sorpresa__']
		dic['acumuladonegative'] = dic['__enojo__'] + dic['__miedo__'] + dic['__repulsion__'] + dic['__tristeza__']
		
		features.append (dic)
	
	
	return features

#emojis

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



def negacion_pln(texto):
    palabras = texto.split()
    negaciones = set(["no", "sin", "ni", "nada", "nunca", "tampoco", "nadie", "jamas", "ninguno"])
    nueva_lista = []
    i = 0
    while i < len(palabras):
        palabra_actual = palabras[i].lower()
        if palabra_actual in negaciones:
            try:
                siguiente = palabras[i+1].lower()

                # Asegurarse de que no haya signos de puntuación antes o después de las palabras
                
                nueva_lista.append(f"{palabra_actual}_{siguiente}")
                i += 1
            except IndexError:
                pass
        else:
            nueva_lista.append(palabras[i])
        i += 1

    return " ".join(nueva_lista)

"""# Create training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    corpus['Opinion'], corpus['Polarity'], test_size=0.2, random_state=0
)"""



import time
start_time = time.time()










corpus = pd.read_excel('Rest_Mex_2022_preprocesado.xlsx')

# Aplica la función de negación
#corpus['Opinion'] = corpus['Opinion'].apply(negacion_pln)

# Combina las columnas 'Title' y 'Opinion' en una nueva columna llamada 'Combined'
corpus['Combined'] = corpus['Title'].astype(str) + ' ' + corpus['Opinion'].astype(str)
# Aplica la función de negación
corpus['Combined'] = corpus['Combined'].apply(negacion_pln)
# Crear conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    corpus['Combined'], corpus['Polarity'], test_size=0.2, random_state=0
)

# Create different text representations of the corpus (example: TF-IDF)

"""
vectorizador = CountVectorizer(binary=True)
#vectorizador = TfidfVectorizer()
X_train_vectorizado = vectorizador.fit_transform(X_train)
X_test_vectorizado = vectorizador.transform(X_test)

"""

from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
print("EMBEDIGNS JEJE")
# Tokenización simple y creación de una lista de listas de palabras
tokenized_data = [simple_preprocess(text) for text in X_train]

# Entrenar el modelo Word2Vec
model = Word2Vec(sentences=tokenized_data, vector_size=2000, window=5, min_count=1, workers=4)

# Construir el vocabulario
model.build_vocab(tokenized_data)

# Entrenar el modelo Word2Vec
model.train(tokenized_data, total_examples=model.corpus_count, epochs=10)

# Función para obtener la representación vectorial de un documento
def get_vector_representation(document, model):
    words = simple_preprocess(document)
    vector = sum(model.wv[word] for word in words if word in model.wv)
    return vector / len(vector) if len(vector) > 0 else [0] * model.vector_size

# Obtener representaciones vectoriales para el conjunto de entrenamiento y prueba
X_train_embeddings = [get_vector_representation(document, model) for document in X_train]
X_test_embeddings = [get_vector_representation(document, model) for document in X_test]

# Convertir las listas a arrays
#X_train_embeddings = pd.DataFrame(X_train_embeddings).values
#X_test_embeddings = pd.DataFrame(X_test_embeddings).values

X_train_embeddings = np.array(X_train_embeddings)
X_test_embeddings = np.array(X_test_embeddings)

X_train_vectorizado = X_train_embeddings
X_test_vectorizado = X_test_embeddings
print(X_train_vectorizado)
#imprimir tamaño de la matriz
print(X_train_vectorizado.shape)

"""X_train_vectorizado = np.nan_to_num(X_train_embeddings)
X_train_vectorizado = np.vstack(X_train_vectorizado)
X_test_vectorizado = np.nan_to_num(X_test_embeddings)
X_test_vectorizado = np.vstack(X_test_vectorizado)"""



"""

#EMOJIS---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# Ruta al archivo Excel
xlsx_file_path = 'Recursos/Emojis lexicon.xlsx'

# Llama a la función para cargar el diccionario
emoji_sentiments_dict = load_emoji_sentiments(xlsx_file_path)

# Imprime el diccionario resultante
print(emoji_sentiments_dict)

# Ajustar el vectorizador y obtener el vocabulario
X_train_vectorizado = vectorizador.fit_transform(X_train)
vocabulary = vectorizador.get_feature_names_out()

# Calcular pesos basados en los valores de emojis asociados
weights = []
for word in vocabulary:
    emoji_negative = sum(float(emoji_sentiments_dict[emoji]['negative']) for emoji in word if emoji in emoji_sentiments_dict)
    emoji_positive = sum(float(emoji_sentiments_dict[emoji]['positive']) for emoji in word if emoji in emoji_sentiments_dict)
    weight = 1 + emoji_positive - emoji_negative
    weights.append(weight)

# Convertir los pesos a un array numpy
weights_array = np.array(weights)

# Aplicar los pesos directamente a la matriz TF-IDF
X_train_vectorizado_weighted = X_train_vectorizado.multiply(weights_array)

# Transformar el conjunto de prueba usando el mismo vectorizador y aplicar los pesos
X_test_vectorizado = vectorizador.transform(X_test)
X_test_vectorizado_weighted = X_test_vectorizado.multiply(weights_array)

X_train_vectorizado = X_train_vectorizado_weighted
X_test_vectorizado = X_test_vectorizado_weighted"""



#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#POLARIDAD CON BIBLIOTECA

# Función para obtener la polaridad de un texto
def obtener_polaridad(texto):
    blob = TextBlob(texto)
    return blob.sentiment.polarity

# Agrega la polaridad a la matriz TF-IDF
polaridades_train = X_train.apply(obtener_polaridad)
polaridades_test = X_test.apply(obtener_polaridad)

# Agrega las polaridades a la matriz TF-IDF
#X_train_vectorizado_with_polarity = hstack((X_train_vectorizado, polaridades_train.values.reshape(-1, 1)))
#X_test_vectorizado_with_polarity = hstack((X_test_vectorizado, polaridades_test.values.reshape(-1, 1)))
X_train_vectorizado_with_polarity = np.concatenate([X_train_embeddings, polaridades_train.values.reshape(-1, 1)], axis=1)
X_test_vectorizado_with_polarity = np.concatenate([X_test_embeddings, polaridades_test.values.reshape(-1, 1)], axis=1)
#Agrega las polaridades a la matriz de embeddings
#X_train_vectorizado_with_polarity = np.concatenate([X_train_embeddings, polaridades_train.values.reshape(-1, 1)], axis=1)
#X_train_vectorizado_with_polarity = np.concatenate([X_test_embeddings, polaridades_test.values.reshape(-1, 1)], axis=1)


X_train_vectorizado = X_train_vectorizado_with_polarity
X_test_vectorizado = X_test_vectorizado_with_polarity

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------


#
print("Iniciar lexicon")
#Polariadad con lexicon
#Load lexicons
lexicon_sel = load_sel()
polarity_train = getSELFeatures(X_train, lexicon_sel)
polarity_test = getSELFeatures(X_test, lexicon_sel)


#ESTO ES CUANDO SE UTILIZA REPRESENTACION NORMAL

# Construir vectores de polaridad para entrenamiento
polarity_train_vectors = np.array([[p['acumuladopositivo'], p['acumuladonegative']] for p in polarity_train])
# Concatenar vectores de polaridad con la representación TF-IDF
print("X_train_vectorizado shape:", X_train_vectorizado.shape)
print("polarity_train_vectors shape:", polarity_train_vectors.shape)
#X_train_combined = hstack([X_train_vectorizado, polarity_train_vectors])
X_train_combined = np.concatenate([X_train_vectorizado, polarity_train_vectors], axis=1)
# Construir vectores de polaridad para prueba
polarity_test_vectors = np.array([[p['acumuladopositivo'], p['acumuladonegative']] for p in polarity_test])
print("X_test_vectorizado shape:", X_test_vectorizado.shape)
print("polarity_test_vectors shape:", polarity_test_vectors.shape)
# Concatenar vectores de polaridad con la representación vectorizada
#X_test_combined = hstack([X_test_vectorizado, polarity_test_vectors])
X_test_combined = np.concatenate([X_test_vectorizado, polarity_test_vectors], axis=1)
# Ahora, X_train_combined y X_test_combined contienen tanto la representación TF-IDF como la información de polaridad.

"""
#ESTO PARA REPRESENTACION EMBEDINGS
polarity_train_vectors = np.array([[p['acumuladopositivo'], p['acumuladonegative']] for p in polarity_train])
polarity_test_vectors = np.array([[p['acumuladopositivo'], p['acumuladonegative']] for p in polarity_test])

# Concatenar vectores de polaridad con la representación de embeddings
X_train_combined = np.concatenate([X_train_embeddings, polarity_train_vectors], axis=1)
X_test_combined = np.concatenate([X_test_embeddings, polarity_test_vectors], axis=1)
"""
X_train_vectorizado = X_train_combined
X_test_vectorizado = X_test_combined
print(X_train_vectorizado.shape)
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------


# Split the training set into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Define models
models = [
    #LogisticRegression(max_iter=10000),
	MLPClassifier(hidden_layer_sizes=(100, ),
                                     activation='tanh',
                                     learning_rate_init=0.01,
                                     max_iter=1000,
									 solver='lbfgs',
                                     validation_fraction=0.2,
                                     ),
	#SVC(kernel='sigmoid', C=1, gamma=1 ,probability=True) no jala
]

# Train different Machine Learning models and calculate average f1 macro
best_model = None
best_f1_macro = 0

for model in models:
    print("jheje")
    pipeline = make_pipeline(StandardScaler(with_mean=False), model)  # Example pipeline with StandardScaler
    scores = cross_validate(pipeline, X_train_vectorizado, y_train, cv=kf, scoring='f1_macro', return_train_score=False,error_score='raise')
    avg_f1_macro = scores['test_score'].mean()

    print(f'{model.__class__.__name__} - Average F1 Macro: {avg_f1_macro}')

    if avg_f1_macro > best_f1_macro:
        best_f1_macro = avg_f1_macro
        best_model = model

# Select the best adjusted model
print(f'\nBest Model: {best_model.__class__.__name__} with F1 Macro: {best_f1_macro}')

# Train the selected model using the full train set
best_model.fit(X_train_vectorizado, y_train)

# Use the trained model to predict opinions of the test set
predictions = best_model.predict(X_test_vectorizado)

# Calculate average f1 macro
avg_f1_macro_test = f1_score(y_test, predictions, average='macro')
print(f'\nAverage F1 Macro on Test Set: {avg_f1_macro_test}')


#alerta de que ya termino

import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#imprimir cuanto tiempo tarda

print("--- %s seconds ---" % (time.time() - start_time))


"""





polaridadCadenas = []
#aqui se debe agregar la representación de polaridad
for i in range(len(X_train)):
	polaridadPos = np.array([polarity_train[i]['acumuladopositivo']])  # Wrap it in a list to add a dimension
	polaridadNeg = np.array([polarity_train[i]['acumuladonegative']])  # Wrap it in a list to add a dimension

	# Asegúrate de que los arrays tengan al menos una dimensión
	polaridadPos = polaridadPos.reshape(-1, 1)
	polaridadNeg = polaridadNeg.reshape(-1, 1)
	polaridadCadena = np.concatenate((polaridadPos, polaridadNeg), axis=0)
	polaridadCadenas.append(polaridadCadena)

polaridadCadenas = np.array(polaridadCadenas)
aewa = hstack([X_train_vectorizado, polaridadCadenas]).toarray()


polaridadCadenas = []
for i in range(len(X_test)):
	polaridadPos = np.array(polarity_test[i]['acumuladopositivo'])
	polaridadNeg = np.array(polarity_test[i]['acumuladonegative'])
	polaridadCadena =  np.concatenate((polaridadPos, polaridadNeg), axis=0)
	polaridadCadenas.append(polaridadCadena)

polaridadCadenas = np.array(polaridadCadenas)
X_test_vectorizado = hstack([X_test,polaridadCadenas]).toarray()



"""