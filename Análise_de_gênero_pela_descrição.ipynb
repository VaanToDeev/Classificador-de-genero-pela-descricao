import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Baixar recursos do NLTK (se necessário)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Adicionado para baixar o recurso 'omw-1.4'

# Dados de treinamento
descriptions = [
    ("Este é um jogo de tiro em primeira pessoa cheio de ação e aventura."),
    ("Neste jogo de estratégia, você precisa construir uma civilização e conquistar territórios."),
    ("Jogue como um detetive e resolva mistérios neste jogo de aventura e investigação."),
    ("Explore mundos abertos, complete missões e personalize seu personagem neste RPG épico."),
    ("Embarque em uma jornada épica para salvar o reino neste jogo de fantasia cheio de magia e monstros.")
]
genres = ["Ação", "Estratégia", "Aventura", "RPG", "Fantasia"]

# Pré-processamento dos dados
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

preprocessed_descriptions = [preprocess_text(description) for description in descriptions]

# Vetorização dos dados de treinamento
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(preprocessed_descriptions)
y_train = genres

# Treinamento do modelo
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Função para classificar uma nova descrição de jogo
def classify_game_genre(description):
    preprocessed_description = preprocess_text(description)
    X_test = vectorizer.transform([preprocessed_description])
    predicted_genre = classifier.predict(X_test)[0]
    return predicted_genre

# Exemplo de uso
new_description = "Evolua sua civilização para alcançar a gloria."
predicted_genre = classify_game_genre(new_description)
print("Gênero do jogo:", predicted_genre)
