import pandas as pd
import gensim
from gensim import corpora
from gensim.corpora import MmCorpus
from gensim.models import LdaMulticore, CoherenceModel
import matplotlib.pyplot as plt
import numpy as np
from pythainlp.corpus import thai_stopwords
import re
from tqdm import tqdm
tqdm.pandas()

# Load Thai stopwords
thai_stopwords = list(thai_stopwords())

def clean_tokens(tokens):
    cleaned_tokens = []
    for token in tokens:
        # Remove English words
        if re.search(r'[a-zA-Z]', token):
            continue
        # Remove digits
        if re.search(r'\d', token):
            continue
        # Remove Thai stopwords
        if token in thai_stopwords:
            continue
        # Remove short words (length <= 2)
        if len(token) <= 2:
            continue
        # Remove symbols/punctuation
        if re.search(r'[^\u0E00-\u0E7F]', token):  # Keep only Thai characters
            continue
        cleaned_tokens.append(token)
    return cleaned_tokens

# Load data
print("Loading data...")
df = pd.read_feather('final_lda_dataset.feather')  # Replace with your file path
texts = df['tokenized_text_attacut'].tolist()

# Clean tokenized text
print("Cleaning tokenized data...")
df['cleaned_tokens'] = df['tokenized_text_attacut'].progress_apply(clean_tokens)

# Remove empty documents after cleaning
df = df[df['cleaned_tokens'].progress_apply(len) > 0]

print(f"Filtered to {len(df)} documents after cleaning")

# Create dictionary and filter terms
print("Creating dictionary and corpus...")
texts = df['cleaned_tokens'].tolist()
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=20, no_above=0.5)  # Adjust these parameters

# Create filtered corpus and aligned texts
corpus = []
texts_filtered = []
for text in texts:
    bow = dictionary.doc2bow(text)
    if len(bow) > 0:
        corpus.append(bow)
        texts_filtered.append(text)

print(f"Filtered to {len(corpus)} documents after removing empty documents")

# Grid search parameters
topic_nums = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
coherence_scores = []

# Perform grid search
print("\nStarting grid search...")
for num_topics in topic_nums:
    print(f"\nTraining model with {num_topics} topics...")
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics,
                             workers=4,
                             passes=10,
                             iterations=50,
                             random_state=42,
                             alpha='symmetric')
    
    # Calculate coherence
    coherence_model = CoherenceModel(model=lda_model,
                                    texts=texts_filtered,
                                    dictionary=dictionary,
                                    coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append(coherence_score)
    
    print(f"Coherence score for {num_topics} topics: {coherence_score:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(topic_nums, coherence_scores, marker='o', linestyle='--')
plt.title('Topic Model Coherence Scores')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.grid(True)
plt.savefig('coherence_scores_default.png')
plt.show()

# Determine best number of topics
best_idx = np.argmax(coherence_scores)
best_num_topics = topic_nums[best_idx]
print(f"\nOptimal number of topics: {best_num_topics} (Coherence: {coherence_scores[best_idx]:.4f})")

# Train and save final model
print("\nTraining final model...")
final_model = LdaMulticore(corpus=corpus,
                          id2word=dictionary,
                          num_topics=best_num_topics,
                          workers=4,
                          passes=20,  # More passes for final model
                          iterations=100,
                          random_state=42,
                          alpha='symmetric')
final_model.save('best_lda_default_model/best_lda_model.model')

# Show topics
print("\nTop terms per topic:")
topics = final_model.print_topics(num_words=10)
for topic in topics:
    print(f"\nTopic {topic[0]}:")
    print(topic[1])
    
# Save dictionary
print("Saving dictionary...")
dictionary.save('best_lda_default_model/dictionary.dict')

# Save corpus
print("Saving corpus...")
MmCorpus.serialize('best_lda_default_model/corpus.mm', corpus)

# Save cleaned data
print("Saving cleaned data...")
df.to_feather('cleaned_data.feather')