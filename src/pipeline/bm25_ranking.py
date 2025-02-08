from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import time


from data import read_jsonl, write_file
from functions import write_file, evaluate_dataset
from collections import defaultdict


def preprocess(text, remove_stopwords=True, stemming=True, lemmatization=True):
    # 1. 소문자 변환
    text = text.lower()
    # 2. 문장 부호 제거
    text = "".join([char for char in text if char not in string.punctuation])
    # 3. 토큰화
    tokens = word_tokenize(text)
    # 4. 불용어 제거
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]
    # 5. 어간 추출 또는 표제어 추출
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    elif lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = tokens[: min(256, len(tokens))]
    return tokens


def get_bm25(documents):
    doc_ids = [doc["doc_id"] for doc in documents]
    doc_texts = [doc["text"] for doc in documents]
    processed_docs = [preprocess(doc_text) for doc_text in doc_texts]

    bm25 = BM25Okapi(processed_docs)
    return bm25, doc_ids


def get_top_k_documents(query, bm25, doc_ids, k=10):
    query_tokens = preprocess(query["query"])
    scores = bm25.get_scores(query_tokens)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :k
    ]
    top_k_doc_ids = [doc_ids[i] for i in top_k_indices]
    return top_k_doc_ids


# python -c "import nltk; nltk.download('stopwords')"
# python -c "import nltk; nltk.download('punkt')"
def do_expermient(query_path, doc_path, session_number):
    query_data = read_jsonl(query_path)[:100]
    doc_data = read_jsonl(doc_path)[:100]

    query_count = len(query_data)
    doc_count = len(doc_data)
    print(f"#{session_number} | Query count:{query_count}, Document count:{doc_count}")

    bm25, doc_ids = get_bm25(doc_data)
    qcnt = 0
    result = defaultdict(list)

    for qidx in range(query_count):
        query = query_data[qidx]
        qid = query["qid"]
        top_k_doc_ids = get_top_k_documents(query, bm25, doc_ids, k=10)
        result[qid].extend(top_k_doc_ids)

        qcnt += 1
        if qcnt % 10 == 0:
            print(f"qcnt: {qcnt}")

    rankings_path = f"../data/rankings/bm25-{session_number}.txt"
    write_file(rankings_path, result)
    evaluate_dataset(query_path, rankings_path, doc_count)


def evaluate(sesison_count=1):
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = f"../data/sessions/test_session{session_number}_queries.jsonl"
        eval_doc_path = f"../data/sessions/test_session{session_number}_docs.jsonl"

        start_time = time.time()
        do_expermient(eval_query_path, eval_doc_path, session_number)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")
