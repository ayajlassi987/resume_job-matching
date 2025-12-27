from src.rag_engine import RAGEngine


def run_interview_copilot(resumes, jobs, matches, question_type="default"):
    """
    Run interview copilot to generate questions for matched candidates.
    
    Args:
        resumes: List of resume texts
        jobs: List of job description texts
        matches: List of job indices for each resume
        question_type: Type of questions to generate (technical, behavioral, system_design, default)
    """
    rag = RAGEngine()

    for i, job_idxs in enumerate(matches[:3]):
        if not job_idxs:
            continue
        job_idx = job_idxs[0]

        questions = rag.generate_questions(
            resumes[i],
            jobs[job_idx],
            question_type=question_type
        )

        print(f"\nCandidate {i} - Job {job_idx}")
        print(f"Question Type: {question_type}")
        for j, q in enumerate(questions, 1):
            print(f"{j}. {q}")
