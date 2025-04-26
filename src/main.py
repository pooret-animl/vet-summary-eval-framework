from grader import PDFSummaryGrader

google_api_key = "ASDRQVASdgfaeraVASFA"
summaries_path = "summary_path"
grader = PDFSummaryGrader(api_key = google_api_key)
results = grader.create_grading_report(summaries_path)

if __name__ == "__main__":
    pass