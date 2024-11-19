from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import re
import numpy as np
import faiss
import time
from collections import Counter
import string



# Configure the generative model
genai.configure(api_key='AIzaSyDQv2SUMMoqgE_vtEp1WWpr_XVyA09kcDM')
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Database setup
DATABASE = 'chat_app.db'

def extract_keywords(text):
    # Convert to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))

    # Split into words
    words = text.split()

    # Count word frequencies
    word_counts = Counter(words)

    # Get the top 3 most common words
    most_common_words = [word for word, _ in word_counts.most_common(3)]

    # Join them to form a session name
    return " ".join(most_common_words).capitalize()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    with conn:
        # conn.execute('''DROP TABLE IF EXISTS CHAT''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS person (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS session (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_name TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES person(id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES session(id)
            )
        ''')

        
    conn.close()




# Initialize the database
init_db()



# Load documents from erp.txt
def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    documents = content.split('\n\n')  # Split based on double newlines
    return documents


# Assuming we use a 768-dimensional embedding space (e.g., from BERT)
dimension = 768
index = faiss.IndexFlatL2(dimension)

documents = load_documents('erp.txt')
embeddings = []


# Function to generate embeddings (replace with actual model)
def generate_embeddings(documents):
    for doc in documents:
        # Placeholder: Replace with actual embedding generation (e.g., BERT)
        embedding = np.random.rand(dimension).astype('float32')
        embeddings.append(embedding)
        index.add(np.array([embedding]))  # Add to FAISS index


generate_embeddings(documents)


def clean_response(response_text):
    # Clean and format the response text
    response_text = re.sub(r'\d+\.', r'<br>', response_text)
    response_text = re.sub(r'\*\*(.*?):', r'<br><br><strong>\1</strong><br>', response_text)
    response_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response_text)
    response_text = re.sub(r'(:)', r':<br>', response_text)
    response_text = re.sub(r'[*#]', '', response_text)
    return response_text


def generate_query_embedding(text):
    # Placeholder: Replace with actual embedding generation for the query
    return np.random.rand(dimension).astype('float32')


def generate_response_with_backoff(input_text, retries=5, backoff_factor=2):
    for attempt in range(retries):
        try:
            # Generate a response using the Gemini model
            response = model.generate_content(input_text)
            return response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response."
        except:
            if attempt < retries - 1:
                time.sleep(backoff_factor ** attempt)
            else:
                return "Sorry, the system is currently overloaded. Please try again later."


@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        conn.execute('INSERT INTO person (username, password) VALUES (?, ?)',
                     (username, hashed_password))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM person WHERE username = ?', (username,)).fetchone()


        conn.close()

        # save session_id using select

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']

            return redirect(url_for('home'))
        else:
            return 'Invalid username or password'

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


def is_relevant(query_embedding, retrieved_embeddings, threshold=0.5):
    # Calculate cosine similarity between query embedding and retrieved document embeddings
    similarities = cosine_similarity([query_embedding], retrieved_embeddings)

    # Check if any similarity exceeds the threshold
    return np.any(similarities >= threshold)


@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'response': 'Error: User not logged in'}), 401

    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({'response': 'Error: No message provided'}), 400

    # Generate query embedding
    query_embedding = generate_query_embedding(user_message)

    # Perform a search in FAISS index
    D, I = index.search(np.array([query_embedding]), k=3)  # Retrieve top 3 results
    retrieved_docs = [documents[i] for i in I[0]]
    retrieved_embeddings = [embeddings[i] for i in I[0]]

    # Determine if the retrieved documents are relevant using semantic similarity
    if is_relevant(query_embedding, retrieved_embeddings, threshold=0.5):
        # Integrate relevant document information with the user's message
        context_summary = " ".join(retrieved_docs)
        refined_input = f"User message: {user_message}\n\nContext: {context_summary}"

        # Generate a response using the refined input
        bot_response = generate_response_with_backoff(refined_input)
    else:
        # Generate a general response using the user's message alone
        bot_response = generate_response_with_backoff(user_message)

    cleaned_response = clean_response(bot_response)

    # Save user and bot messages to the database
    conn = get_db_connection()


    conn.execute('INSERT INTO chat (session_id, message, response) VALUES (?, ?, ?)',
                 # use session_id
                 (session['u_id'], user_message, cleaned_response))
    conn.commit()
    conn.close()

    session_name = extract_keywords(cleaned_response)

    # conn.execute('UPDATE session SET session_name = ? WHERE user_id = ?',
    #              (session_name, session['u_id']))

    return jsonify({'response': cleaned_response})


@app.route('/new_chat', methods=['POST'])
def new_chat():
    if 'user_id' not in session:
        return jsonify({'response': 'Error: User not logged in'}), 401

    conn = get_db_connection()
    conn.execute('INSERT INTO session (user_id, session_name) VALUES (?, ?)',
                 (session['user_id'], 'New Chat'))

    sess = conn.execute('SELECT * FROM session WHERE user_id = ?', (session['user_id'],)).fetchone()


    conn.commit()
    chat_session_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]

    session['u_id'] = chat_session_id
    print(session['u_id'])

    conn.close()

    return jsonify({'chat_session_id': chat_session_id, 'session_name': 'New Chat'})

@app.route('/get_chats', methods=['GET'])
def get_chats():
    if 'user_id' not in session:
        return jsonify({'response': 'Error: User not logged in'}), 401

    conn = get_db_connection()

    # Fetch chat sessions for the logged-in user
    chat_sessions = conn.execute('SELECT id, session_name FROM session WHERE user_id = ?',
                                 (session['user_id'],)).fetchall()

    chat_list = []

    for chat_session in chat_sessions:
        print(f"Fetching messages for session_id: {chat_session['id']}")  # Debug statement

        # Fetch messages and responses for each session
        messages = conn.execute('SELECT message, response FROM chat WHERE session_id = ?',
                                (chat_session['id'],)).fetchall()

        print(f"Messages fetched: {messages}")  # Debug statement

        chat_messages = [{'message': msg['message'], 'response': msg['response']} for msg in messages]

        print(f"Processed chat_messages: {chat_messages}")  # Debug statement

        chat_list.append({
            'id': chat_session['id'],
            'session_name': chat_session['session_name'],
            'messages': chat_messages
        })

    print(chat_list)  # Final debug statement
    conn.close()

    return jsonify({'chats': chat_list})



if __name__ == '__main__':
    app.run(debug=True)
