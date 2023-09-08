#!/usr/bin/env python
# coding: utf-8

# In[171]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# # Baixar recursos do NLTK (se necessário)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Adicionado para baixar o recurso 'omw-1.4'

# In[153]:


descricao = [
    "Lidere uma equipe de elite em missões secretas de combate ao terrorismo em locais perigosos ao redor do mundo.",
    "Entre em uma guerra futurista como um cavaleiro da vanguarda equipado com armaduras avançadas e tecnologia de ponta.",
    "Corra contra o tempo e outros caçadores em busca de um tesouro escondido no deserto, enfrentando armadilhas e inimigos mortais.",
    "Lute pela sobrevivência em uma cidade tomada por zumbis, buscando recursos e procurando por outros sobreviventes.",
    "Assuma o papel de um agente secreto em uma busca implacável por vingança contra uma organização criminosa.",
    "Lidere uma equipe de resgate em uma montanha coberta de neve, enfrentando avalanches e perigos naturais.",
    "Enfrente um exército de máquinas autônomas em um mundo onde a inteligência artificial se revoltou.",
    "Torne-se um assassino profissional e navegue pelo submundo do crime em busca de alvos de alto perfil.",
    "Explore planetas desconhecidos, lute contra alienígenas e colete recursos para a expansão da humanidade no espaço.",
    "Participe de uma operação de resgate em uma ilha isolada, enfrentando terroristas e salvando reféns.",
    "Em um mundo infestado de monstros, aceite contratos para caçar e eliminar as ameaças sobrenaturais.",
    "Planeje e execute roubos audaciosos, escapando da polícia em perseguições em alta velocidade.",
    "Lute pela sobrevivência em uma selva hostil, enfrentando predadores e elementos naturais.",
    "Pilote um mecha gigante para combater robôs destrutivos que ameaçam a cidade.",
    "Seja um atirador de elite e complete missões de sniper em cenários urbanos e rurais.",
    "Monte uma equipe especializada para realizar assaltos a bancos meticulosamente planejados.",
    "Em uma recriação do Velho Oeste, torne-se um pistoleiro lendário em duelos e missões emocionantes.",
    "Lute em uma arena de gladiadores contra adversários formidáveis para ganhar sua liberdade.",
    "Una forças com outros caçadores de alienígenas para proteger a Terra de uma invasão de seres de outro mundo.",
    "Viaje no tempo e participe de batalhas históricas em diferentes eras para preservar a linha do tempo.",
    "Em um mundo de alta fantasia, você é o escolhido para enfrentar um antigo mal que ameaça a existência. Viaje por reinos épicos, reúna aliados e domine magias poderosas para derrotar o mal supremo.",
    "Navegue pelos mares perigosos como um temido pirata em busca de tesouros lendários. Recrute uma tripulação leal, melhore seu navio e desafie outros capitães em duelos navais emocionantes.",
    "Assuma o papel de um detetive em uma cidade cheia de mistérios e crimes. Resolva casos complexos, colete evidências e tome decisões que afetam o destino da cidade e de seus habitantes.",
    "Explore uma galáxia vasta como um caçador de recompensas em busca de alvos procurados. Escolha seu próprio caminho, seja um herói ou vilão, e decida o destino de planetas inteiros.",
    "Viaje para um mundo mágico de criaturas mitológicas e deuses antigos. Participe de uma busca épica para restaurar a ordem divina e impedir que o caos consuma tudo.",
    "Seja um aventureiro em um reino de dragões e magia. Enfrente dragões lendários, descubra artefatos mágicos e escolha entre diferentes classes para forjar seu próprio destino.",
    "Em um cenário pós-apocalíptico, lute pela sobrevivência em um mundo devastado. Junte-se a uma facção, construa abrigos e enfrente ameaças mutantes em busca de recursos escassos.",
    "Desbrave uma floresta misteriosa como um caçador de monstros contratado. Rastreie bestas terríveis, aprenda suas fraquezas e proteja aldeias indefesas contra o mal.",
    "Seja um mago em uma academia de magia, onde você pode aprender feitiços, criar poções e enfrentar criaturas mágicas. Descubra segredos antigos e desvende conspirações obscuras.",
    "Assuma o papel de um líder de uma civilização em expansão. Tome decisões políticas, construa cidades e leve sua nação à grandeza através da diplomacia ou da conquista.",
    "Em um mundo cyberpunk, mergulhe em uma trama complexa de intrigas corporativas. Hackeie sistemas, escolha seu alinhamento moral e lute contra megacorporações corruptas.",
    "Viaje para o Egito Antigo e torne-se um explorador de tumbas em busca de tesouros escondidos. Descubra segredos ancestrais e supere armadilhas mortais.",
    "Seja um herói em um reino de contos de fadas, onde você pode se tornar um cavaleiro nobre ou um feiticeiro malévolo. Tome decisões morais que moldam sua jornada.",
    "Navegue pelos céus em um dirigível a vapor em um mundo steampunk. Explore cidades flutuantes, combata piratas do ar e descubra segredos tecnológicos.",
    "Assuma o papel de um agente secreto em missões de espionagem internacional. Use disfarces, gadgets e estratégias para desvendar conspirações globais.",
    "Entre em um mundo de mitos nórdicos e deuses poderosos. Lute contra criaturas lendárias, forje alianças divinas e prove seu valor nos salões de Valhala.",
    "Seja um cientista em um laboratório de pesquisa de alta tecnologia. Descubra avanços científicos, resolva enigmas e decida como usar novas descobertas para o bem ou para o mal.",
    "Assuma o papel de um detetive paranormal em uma cidade assombrada por eventos sobrenaturais. Investigue fenômenos inexplicáveis e descubra a verdade por trás dos mistérios.",
    "Em um mundo de animais antropomórficos, viva uma vida de fazendeiro em uma comunidade acolhedora. Plante, cultive e construa relacionamentos em um cenário encantador.",
    "Explore uma cidade subterrânea habitada por gnomos e criaturas subterrâneas. Aventure-se em masmorras, colete tesouros e desvende segredos esquecidos.",
    "Construa um império do zero em um mundo medieval. Gerencie recursos, treine exércitos e conquiste territórios enquanto luta pelo domínio total.",
    "Lidere uma civilização através das eras, desde a antiguidade até a era espacial. Tome decisões políticas, faça avanços tecnológicos e defenda sua nação contra ameaças externas.",
    "Em um planeta alienígena hostil, estabeleça uma colônia e lute contra criaturas extraterrestres. Gerencie recursos escassos e construa uma sociedade sustentável.",
    "Assuma o papel de um comandante militar em uma guerra futurista. Planeje estratégias, desenvolva táticas avançadas e lute em batalhas táticas em tempo real.",
    "Em uma ilha tropical paradisíaca, construa um resort de luxo e atraia turistas de todo o mundo. Gerencie hotéis, praias e atrações para alcançar o sucesso.",
    "Comande uma frota de navios em alto mar durante a era da pirataria. Saqueie cidades costeiras, encontre tesouros enterrados e enfrente navios da Marinha Real.",
    "No espaço sideral, seja o capitão de uma nave interestelar. Explore sistemas solares, faça negócios com alienígenas e participe de batalhas espaciais estratégicas.",
    "Assuma o controle de um reino de fantasia e expanda suas fronteiras através de conquistas e diplomacia. Construa cidades, treine exércitos e conduza sua nação à glória.",
    "Em um mundo pós-apocalíptico, lidere um grupo de sobreviventes em busca de um refúgio seguro. Tome decisões difíceis, fortifique sua base e enfrente gangues rivais.",
    "Construa um parque temático espetacular, desde montanhas-russas até zoológicos exóticos. Atraia visitantes, mantenha a segurança e faça seu parque prosperar.",
    "Na Roma Antiga, torne-se um estrategista militar e comande legiões poderosas. Conquiste territórios, fortaleça sua posição e forje o destino de um império.",
    "Em um ambiente de guerra moderna, assuma o controle de um país e tome decisões críticas em conflitos globais. Gerencie recursos, negocie alianças e evite desastres nucleares.",
    "Em um mundo mágico, monte uma guilda de aventureiros e embarque em missões épicas. Desenvolva estratégias táticas para derrotar monstros lendários.",
    "Construa uma cidade do futuro em um ambiente de ficção científica. Gerencie infraestrutura, resolva problemas sociais e faça avanços tecnológicos.",
    "Lidere uma nação de vikings em incursões marítimas e raids. Planeje ataques a vilas costeiras, saqueie tesouros e ganhe renome como um líder destemido.",
    "No coração da Segunda Guerra Mundial, comande tropas em batalhas históricas. Desenvolva estratégias militares, conduza operações secretas e vença a guerra.",
    "Assuma o controle de um time de esportes e leve-o à vitória em ligas e campeonatos. Gerencie treinamentos, contratações e estratégias de jogo para alcançar o topo.",
    "Em uma cidade cyberpunk, lidere uma gangue de rua e conquiste territórios em uma luta pelo poder. Use táticas urbanas, implantes cibernéticos e influência política.",
    "No cenário político da Guerra Fria, seja um espião e participe de operações secretas de espionagem. Engane agentes duplos, decifre códigos e evite o apocalipse nuclear.",
    "Em um mundo de fantasia sombria, construa um exército de criaturas mágicas e comande-o em batalhas estratégicas contra forças malignas. Domine a magia e salve o reino.",
    "Torne-se um mestre das artes marciais em um jogo de aventura de ação repleto de combates intensos e desafios de treinamento.",
    "Explore um mundo dividido em quatro reinos elementares, cada um com seus próprios desafios e segredos, enquanto busca restaurar o equilíbrio.",
    "Assuma o papel de um bibliotecário em uma biblioteca mágica escondida e resolva enigmas para desvendar segredos antigos.",
    "Parta em uma aventura marítima em busca de tesouros perdidos e enfrente tripulações fantasmas em navios assombrados.",
    "Testemunhe uma batalha épica entre deuses de diferentes panteões e escolha seu lado nesse conflito divino.",
    "Aprenda os segredos da alquimia enquanto viaja pelo mundo em busca de ingredientes raros e realiza experimentos mágicos.",
    "Navegue por um labirinto temporal em constante mudança, enfrentando paradoxos e desafios temporais.",
    "Junte-se a uma ordem de guardiões e proteja um templo antigo de forças malignas que buscam um artefato sagrado.",
    "Explore a galáxia em busca de planetas desconhecidos, vida alienígena e segredos cósmicos.",
    "Aventure-se em ruínas amaldiçoadas em busca de uma cura para uma terrível maldição que afeta sua equipe de exploração.",
    "Em um mundo onde a magia está desaparecendo, embarque em uma missão para salvar essa fonte de poder.",
    "Planeje uma fuga ousada de uma prisão de alta segurança, evitando guardas e resolvendo quebra-cabeças para alcançar a liberdade.",
    "Explore dois reinos opostos, o Paraíso e o Inferno, enquanto enfrenta dilemas morais e decide o destino de sua alma.",
    "Siga os passos do lendário Rei Arthur e seus cavaleiros da Távola Redonda em busca do Santo Graal.",
    "Investigue uma estação espacial abandonada cheia de segredos sombrios e criaturas desconhecidas.",
    "Torne-se um herói lendário e empunhe uma espada mágica para enfrentar monstros e vilões temíveis.",
    "Viaje ao Egito antigo e desvende os segredos das pirâmides enquanto enfrenta armadilhas mortais.",
    "Proteja um refúgio secreto para viajantes do tempo e evite mudanças na linha do tempo.",
    "Embarque em uma aventura épica para encontrar o elixir da imortalidade, enfrentando desafios sobrenaturais e caçadores de tesouros.",
    "Explore um mundo surreal de sonhos e pesadelos, resolvendo quebra-cabeças para encontrar uma saída."
]


# In[154]:


generos = ['ação'] * 20 + ['RPG'] * 20 + ['estrategia'] * 20 + ['aventura'] * 20 


# In[155]:


# Dados de treinamento
descriptions = descricao
    
genres = generos


# In[156]:


# Pré-processamento dos dados
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[157]:


def preprocess_text(text):
    text = str(text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

preprocessed_descriptions = [preprocess_text(description) for description in descriptions]


# In[158]:


X_train, X_test, y_train, y_test = train_test_split(preprocessed_descriptions, generos, test_size=0.2, random_state=42)


# In[159]:


preprocessed_descriptions


# In[160]:


# Vetorização dos dados de treinamento
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# In[161]:


# Treinamento do modelo
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[162]:


# Faça previsões no conjunto de teste
y_pred = classifier.predict(X_test)


# In[163]:


report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[164]:


# Função para classificar uma nova descrição de jogo
def classify_game_genre(description):
    preprocessed_description = preprocess_text(description)
    X_test = vectorizer.transform([preprocessed_description])
    predicted_genre = classifier.predict(X_test)[0]
    return predicted_genre


# In[165]:


# Exemplo de uso
new_description = input('digite a descrição')
predicted_genre = classify_game_genre(new_description)
print("Gênero do jogo:", predicted_genre)


# In[166]:


#gráfico de pizza
plt.figure(figsize=(8, 8))
plt.title("Distribuição dos Gêneros")
y_train_series = pd.Series(y_train)
y_train_series.value_counts().plot.pie(autopct="%1.1f%%")
plt.show()


# In[172]:


precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
accuracy = accuracy_score(y_test, y_pred)

metrics = ["Precision", "Recall", "F1-Score", "accuracy"]
values = [precision, recall, f1, accuracy]

plt.figure(figsize=(10, 6))
sns.barplot(x=metrics, y=values, palette="Blues")
plt.title("Métricas de Avaliação do Modelo")
plt.ylim(0, 1)  # Define o limite do eixo y de 0 a 1 para representar porcentagens
plt.ylabel("Valor")
plt.show()


# In[ ]:




