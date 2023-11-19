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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import spacy
import re
import numpy as np
from scipy.sparse import hstack
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



# Lee el archivo Excel
corpus = pd.read_excel('Rest_Mex_2022.xlsx')


# Create training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    corpus['Opinion'], corpus['Polarity'], test_size=0.2, random_state=0
)
print("Iniciar lexicon")

#Polariadad con lexicon
#Load lexicons
lexicon_sel = load_sel()
polarity_train = getSELFeatures(X_train, lexicon_sel)
polarity_test = getSELFeatures(X_test, lexicon_sel)
	


# Create different text representations of the corpus (example: TF-IDF)
tfidf_vectorizer = CountVectorizer(binary=True)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print ((X_train_tfidf))

# Construir vectores de polaridad para entrenamiento
polarity_train_vectors = np.array([[p['acumuladopositivo'], p['acumuladonegative']] for p in polarity_train])
# Concatenar vectores de polaridad con la representación TF-IDF
X_train_combined = hstack([X_train_tfidf, polarity_train_vectors])

# Construir vectores de polaridad para prueba
polarity_test_vectors = np.array([[p['acumuladopositivo'], p['acumuladonegative']] for p in polarity_test])
# Concatenar vectores de polaridad con la representación TF-IDF
X_test_combined = hstack([X_test_tfidf, polarity_test_vectors])
# Ahora, X_train_combined y X_test_combined contienen tanto la representación TF-IDF como la información de polaridad.

X_train_tfidf = X_train_combined
X_test_tfidf = X_test_combined

print ((X_train_tfidf))

# Split the training set into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Define models
models = [
    MultinomialNB(),
    LogisticRegression(max_iter=10000),
]

# Train different Machine Learning models and calculate average f1 macro
best_model = None
best_f1_macro = 0

for model in models:
    print("jheje")
    pipeline = make_pipeline(StandardScaler(with_mean=False), model)  # Example pipeline with StandardScaler
    scores = cross_validate(pipeline, X_train_tfidf, y_train, cv=kf, scoring='f1_macro', return_train_score=False)
    avg_f1_macro = scores['test_score'].mean()

    print(f'{model.__class__.__name__} - Average F1 Macro: {avg_f1_macro}')

    if avg_f1_macro > best_f1_macro:
        best_f1_macro = avg_f1_macro
        best_model = model

# Select the best adjusted model
print(f'\nBest Model: {best_model.__class__.__name__} with F1 Macro: {best_f1_macro}')

# Train the selected model using the full train set
best_model.fit(X_train_tfidf, y_train)

# Use the trained model to predict opinions of the test set
predictions = best_model.predict(X_test_tfidf)

# Calculate average f1 macro
avg_f1_macro_test = f1_score(y_test, predictions, average='macro')
print(f'\nAverage F1 Macro on Test Set: {avg_f1_macro_test}')





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
aewa = hstack([X_train_tfidf, polaridadCadenas]).toarray()


polaridadCadenas = []
for i in range(len(X_test)):
	polaridadPos = np.array(polarity_test[i]['acumuladopositivo'])
	polaridadNeg = np.array(polarity_test[i]['acumuladonegative'])
	polaridadCadena =  np.concatenate((polaridadPos, polaridadNeg), axis=0)
	polaridadCadenas.append(polaridadCadena)

polaridadCadenas = np.array(polaridadCadenas)
X_test_tfidf = hstack([X_test,polaridadCadenas]).toarray()



"""