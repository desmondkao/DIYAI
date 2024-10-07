import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import numpy as np
import re
import os


# Function to preprocess and tokenize text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])  # Fit tokenizer on text
    total_words = len(tokenizer.word_index) + 1  # Get total number of unique words

    token_list = tokenizer.texts_to_sequences([text])[0]  # Convert text to sequence of tokens
    max_sequence_len = 100  # Increase sequence length
    input_sequences = create_sequences(token_list, max_sequence_len)  # Create input sequences

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))  # Pad sequences

    X, y = input_sequences[:, :-1], input_sequences[:, -1]  # Split input sequences into X and y
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)  # One-hot encode y

    return tokenizer, total_words, max_sequence_len, X, y


def create_sequences(token_list, max_sequence_len):
    input_sequences = []
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        if len(n_gram_sequence) <= max_sequence_len:
            input_sequences.append(n_gram_sequence)
    return input_sequences


# Function to build the model
def build_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=300))  # Add embedding layer
    model.add(Bidirectional(LSTM(256, return_sequences=True)))  # Add bidirectional LSTM layer
    model.add(Dropout(0.3))  # Add dropout layer
    model.add(Bidirectional(LSTM(256, return_sequences=True)))  # Add bidirectional LSTM layer
    model.add(Dropout(0.3))  # Add dropout layer
    model.add(Bidirectional(LSTM(128)))  # Add bidirectional LSTM layer
    model.add(Dense(256, activation='relu'))  # Add dense layer
    model.add(Dense(total_words, activation='softmax'))  # Add output layer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Set optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # Compile model
    model.summary()  # Print model summary
    return model


# Function for Top-k sampling
def top_k_sampling(predictions, k=10):
    sorted_indices = np.argsort(predictions)[::-1][:k]  # Get top-k indices
    top_k_probabilities = predictions[sorted_indices]
    top_k_probabilities = top_k_probabilities / np.sum(top_k_probabilities)  # Normalize probabilities
    predicted_index = np.random.choice(sorted_indices, p=top_k_probabilities)
    return predicted_index


# Function for Top-p (nucleus) sampling
def top_p_sampling(predictions, p=0.9):
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_probabilities = predictions[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probabilities)
    
    # Only include tokens with cumulative probability less than p
    indices_to_keep = sorted_indices[cumulative_probs <= p]
    probabilities_to_keep = sorted_probabilities[:len(indices_to_keep)]
    probabilities_to_keep = probabilities_to_keep / np.sum(probabilities_to_keep)  # Normalize
    
    predicted_index = np.random.choice(indices_to_keep, p=probabilities_to_keep)
    return predicted_index


# Function to generate text
def generate_text(model, tokenizer, seed_text, next_words, max_sequence_len, temperature=0.7, top_k=None, top_p=None):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]  # Convert seed text to sequence of tokens
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')  # Pad sequence
        predictions = model.predict(token_list, verbose=0)[0]  # Predict next word probabilities

        # Apply temperature
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        # Apply Top-k or Top-p sampling
        if top_k is not None:
            predicted_index = top_k_sampling(predictions, k=top_k)
        elif top_p is not None:
            predicted_index = top_p_sampling(predictions, p=top_p)
        else:
            # Default sampling
            predicted_index = np.random.choice(len(predictions), p=predictions)

        predicted_word = tokenizer.index_word.get(predicted_index, '')  # Get predicted word
        if predicted_word:
            seed_text += " " + predicted_word  # Append predicted word to seed text
        else:
            break
    return seed_text


# Read the source text from a file in chunks
def read_source_text_in_chunks(file_path, chunk_size=1024):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)  # Read chunk of text
            if not chunk:
                break
            yield chunk


# Function to append new text to source file
def append_to_source_file(file_path, new_text):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f"\n{new_text}")


# Main execution
if __name__ == "__main__":
    model_path = 'trained_model.h5'

    # Collect all chunks into a single string for processing
    source_text = ""
    for chunk in read_source_text_in_chunks('source_text.txt'):
        source_text += chunk

    # Ask the user if they want to add new text to the source file
    add_text = input("Do you want to add new text to the source corpus and retrain the model? (yes/no): ").lower()
    
    if add_text == "yes":
        new_text = input("Enter the new text to add: ")
        append_to_source_file('source_text.txt', new_text)  # Append the new text to the source file
        source_text += "\n" + new_text  # Update the source text with the new text

        # Preprocess the source text
        tokenizer, total_words, max_sequence_len, X, y = preprocess_text(source_text)

        # Build the model
        model = build_model(total_words, max_sequence_len)

        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        model.fit(X, y, epochs=100, batch_size=64, callbacks=[early_stopping], verbose=1)

        # Save the model
        model.save(model_path)
    else:
        # Load the pre-trained model
        if os.path.exists(model_path):
            model = load_model(model_path)
            tokenizer, total_words, max_sequence_len, X, y = preprocess_text(source_text)
        else:
            print("No pre-trained model found. Please add new text to train the model first.")
            exit()

    # Generate and print the text
    seed_text = input("Enter the seed text: ")
    next_words = int(input("Enter the number of words to generate: "))
    temperature = float(input("Enter the temperature (e.g., 0.7): "))

    # Ask the user for top-k or top-p sampling preferences
    use_top_k = input("Do you want to use Top-k sampling? (yes/no): ").lower() == "yes"
    use_top_p = input("Do you want to use Top-p sampling? (yes/no): ").lower() == "yes"

    if use_top_k:
        top_k = int(input("Enter value for k (e.g., 10): "))
        generated_text = generate_text(model, tokenizer, seed_text, next_words, max_sequence_len, temperature, top_k=top_k)
    elif use_top_p:
        top_p = float(input("Enter value for p (e.g., 0.9): "))
        generated_text = generate_text(model, tokenizer, seed_text, next_words, max_sequence_len, temperature, top_p=top_p)
    else:
        generated_text = generate_text(model, tokenizer, seed_text, next_words, max_sequence_len, temperature)

    print("Generated Text: ", generated_text)
