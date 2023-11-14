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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import spacy

# Load the Rest_Mex_2022 corpus (replace 'your_data.csv' with the actual file name)
corpus = pd.read_excel('Recursos/Rest_Mex_2022.xlsx')

# Apply normalization process (example: lowercase and remove punctuation)
corpus['Opinion'].fillna('', inplace=True)
corpus['Opinion'] = corpus['Opinion'].str.lower()
corpus['Opinion'] = corpus['Opinion'].str.replace('[^\w\s]', '')
#lematizar
nlp = spacy.load("es_core_news_sm")
corpus['Opinion'] = corpus['Opinion'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))
#quitar stopwords
stop_words = [
"el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del", "lo", "este", "ese", "aquel", "estos", "esos", "aquellos", "este", "esta", "estas", "eso", "esa", "esas", "aquello", "alguno", "alguna", "algunos", "algunas",
"a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "en", "entre", "hacia", "hasta", "para", "por", "según", "sin", "so", "sobre", "tras", "durante", "mediante", "excepto", "a través de", "conforme a", "encima de", "debajo de", "frente a", "dentro de",
"y", "o", "pero", "ni", "que", "si", "como", "porque", "aunque", "mientras", "siempre que", "ya que", "pues", "a pesar de que", "además", "sin embargo", "así que", "por lo tanto", "por lo que", "tan pronto como", "a medida que", "tanto como", "no solo... sino también", "o bien", "bien... bien",
"yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "usted", "nosotras", "me", "te", "le", "nos", "os", "les", "se", "mí", "ti", "sí", "conmigo", "contigo", "consigo", "mi", "tu", "su", "nuestro", "vuestro", "sus", "mío", "tuyo", "suyo", "nuestro", "vuestro", "suyo"]
corpus['Opinion'] = corpus['Opinion'].apply(lambda x: " ".join([word for word in x.split() if word not in (stop_words)]))



# Create training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    corpus['Opinion'], corpus['Polarity'], test_size=0.2, random_state=0
)

# Create different text representations of the corpus (example: TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Split the training set into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Define models
models = [
    MultinomialNB(),
    RandomForestClassifier(),
    SVC()
]

# Train different Machine Learning models and calculate average f1 macro
best_model = None
best_f1_macro = 0

for model in models:
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


