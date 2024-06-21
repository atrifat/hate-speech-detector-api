import pickle
import datetime

with open('model_voting_partial_best.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer_count_no_stop_words.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

query = [
    "good morning. love today weather",
    "damn you fucking nigger. just die rotten in your backyard"
]

start = datetime.datetime.now()
result = model.predict_proba(vectorizer.transform(query))

print("elapsed time", datetime.datetime.now() - start)
print(result)
