from modules.utils import load_txt_folder
from src.matcher import match_pipeline
from src.interview_copilot import run_interview_copilot


def main():
    print("Loading resumes...")
    resumes = load_txt_folder("data/resumes", limit=50)

    print("Loading job descriptions...")
    jobs = load_txt_folder("data/job_descriptions", limit=50)

    print("Running resume-job matching...")
    matches = match_pipeline(resumes, jobs)

    print("Generating interview questions...")
    run_interview_copilot(resumes, jobs, matches)


if __name__ == "__main__":
    main()
