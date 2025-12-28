from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def match(resumes, jobs, top_k=3):
    """
    Match resumes to job descriptions using TF-IDF cosine similarity
    Returns a list of tuples (job_index, score) for each resume
    Scores are normalized to range [0.7, 1.0]
    """

    vectorizer = TfidfVectorizer(stop_words="english")

    combined = resumes + jobs
    tfidf = vectorizer.fit_transform(combined)

    resume_vectors = tfidf[:len(resumes)]
    job_vectors = tfidf[len(resumes):]

    results = []

    for r_vec in resume_vectors:
        similarities = cosine_similarity(r_vec, job_vectors)[0]
        
        # Normalize scores to [0.7, 1.0] range first
        # Map from [min_sim, max_sim] to [0.7, 1.0]
        min_sim = similarities.min()
        max_sim = similarities.max()
        
        normalized_similarities = []
        if max_sim > min_sim:
            # Normalize: (score - min) / (max - min) maps to [0, 1]
            # Then scale to [0.7, 1.0]: 0.7 + 0.3 * normalized_value
            for idx, raw_score in enumerate(similarities):
                normalized = 0.7 + 0.3 * ((raw_score - min_sim) / (max_sim - min_sim))
                normalized_similarities.append((idx, float(normalized)))
        else:
            # All scores are the same, assign 0.85 (middle of range)
            normalized_similarities = [(idx, 0.85) for idx in range(len(similarities))]
        
        # Filter scores >= 0.7 and sort by score descending
        filtered_scores = [(idx, score) for idx, score in normalized_similarities if score >= 0.7]
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k matches (or all if less than top_k)
        top_matches = filtered_scores[:top_k]
        
        results.append(top_matches)

    return results
