import pickle

class data_set_polarity:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

class data_set_attraction:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		
# Ruta al archivo .pkl
archivo_pkl = 'Recursos\corpus_attraction.pkl'

# Cargar el archivo .pkl
with open(archivo_pkl, 'rb') as archivo:
    contenido = pickle.load(archivo)

# Mostrar el contenido
print(contenido.X_train)
