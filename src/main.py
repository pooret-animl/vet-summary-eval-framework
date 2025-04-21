from grader import PDFSummaryGrader
import json



summaries_path = "summary_path"
graded_file_name = summaries_path.strip(".json").split("/")[-1] # put in class
grader = PDFSummaryGrader(google_api_key) # include .env
with open(summaries_path, 'r') as f:
    summaries_dict = json.load(f)
results = grader.create_grading_report(summaries_dict)
if __name__ == "__main__":