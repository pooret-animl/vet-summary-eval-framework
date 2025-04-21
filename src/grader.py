import os
from typing import Optional, Dict
from pathlib import Path
import pdfplumber
import gc
from pdf2image import convert_from_path 
from PIL import Image 
import json
import traceback
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os 

class Grade(BaseModel):
    factual_accuracy: int = Field(..., description="Score 1-5 for factual accuracy")
    factual_accuracy_feedback: str = Field(..., description="Feedback for factual accuracy")
    completeness: int = Field(..., description="Score 1-5 for completeness")
    completeness_feedback: str = Field(..., description="Feedback for completeness")
    chronological_order: int = Field(..., description="Score 1-5 for chronological order")
    chronological_order_feedback: str = Field(..., description="Feedback for chronological order")
    clinical_relevance: int = Field(..., description="Score 1-5 for clinical relevance")
    clinical_relevance_feedback: str = Field(..., description="Feedback for clinical relevance")
    organization: int = Field(..., description="Score 1-5 for organization")
    organization_feedback: str = Field(..., description="Feedback for organization")
    overall_feedback: str = Field(..., description="Overall feedback text")

class PDFSummaryGrader:
    def __init__(self, google_api_key: Optional[str] = None):
        """
        Initialize the PDF Summary Grader.

        Args:
            google_api_key: Google API Key for Gemini models. Reads from GOOGLE_API_KEY env var if None.
        """
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.google_api_key:
             raise ValueError("Google API Key must be provided either as an argument or via GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=self.google_api_key)
        self.model_name = 'gemini-2.5-pro-preview-03-25' 

        self.grader_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=16384),
                    temperature=0.1,
                    responseMimeType='application/json',
                    responseSchema=Grade
                    )

        self.ocr_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.0
        )

    def _ocr_image(self, image: Image.Image) -> str:
        """Perform OCR on an image using the Gemini vision model via native SDK."""
        prompt = "Extract all text content verbatim from this medical document image. Preserve formatting like tables or lists if possible."
        try:
            contents = [prompt, image] 
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config =self.ocr_config,
            )
            return response.text
        except Exception as e:
            print(f"!!! ERROR during Native SDK generate_content (Vision): {e}", flush=True)
            traceback.print_exc()
            return "" 

    def process_pdf(self, filepath: str) -> str:
            """
            Extract all text from PDF, handle OCR if needed, and return the full text as a single string.
            """
            if not Path(filepath).is_file():
                raise FileNotFoundError(f"PDF file not found: {filepath}")
            
            extracted_pages_text = []
            contains_scanned_pages = False
            try:
                with pdfplumber.open(filepath) as pdf:
                    print(f"PDF has {len(pdf.pages)} pages.")
                    num_scanned_images = 0 
                    for page_number, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 5:
                            extracted_pages_text.append(f"\n--- Page {page_number + 1} ---\n{page_text}")
                        else:
                            contains_scanned_pages = True
                            try:
                                images = convert_from_path(
                                    filepath,
                                    first_page=page_number + 1,
                                    last_page=page_number + 1,
                                    dpi=200
                                )
                                if images:
                                    ocr_text = self._ocr_image(images[0])
                                    num_scanned_images += 1
                                    print(f"Extracted --- Page {page_number + 1} using OCR")
                                    extracted_pages_text.append(f"\n--- Page {page_number + 1} (OCR) ---\n{ocr_text}")
                                    images[0].close()
                                    del images
                                else:
                                    print(f"Warning: Could not convert page {page_number + 1} to image.")
                                    extracted_pages_text.append(f"\n--- Page {page_number + 1} (Image Conversion Failed) ---\n")
                            except Exception as img_err:
                                print(f"Error converting/OCR-ing page {page_number + 1}: {img_err}")
                                extracted_pages_text.append(f"\n--- Page {page_number + 1} (OCR Error) ---\n")
                            finally:
                                gc.collect()
            except Exception as pdf_err:
                print(f"Error processing PDF file {filepath}: {pdf_err}")
                full_text = "\n".join(extracted_pages_text)
                if not full_text.strip():
                    print("No text could be extracted due to PDF processing error.")
                    return ""
                else:
                    print("Returning partially extracted text due to PDF processing error.")
                    return full_text

            full_text = "\n".join(extracted_pages_text)
            if not full_text.strip():
                print("Warning: No text could be extracted from the PDF.")
                return ""

            print(f"Finished extracting text. Total length: {len(full_text)} characters.")
            if contains_scanned_pages:
                print(f"Note: OCR was performed on {num_scanned_images} page(s).")

            gc.collect()
            return full_text


    def grade_summary(self, pdf_path: str, summary: str) -> Dict:
        """
        Grades a given summary against the entire content of a PDF document.
        """
        final_result = {
            'scores': {}, 'feedback': {}, 'average_score': None, 'overall_feedback': "Processing Error",
            'source_text_length': 0, 'pass': False, 'pdf_name': os.path.basename(pdf_path),
            'pdf_full_content': "", 'error_message': None, 'llm_reasoning': None 
        }
        try:
            print(f"\n--- Grading Summary for PDF: {pdf_path} ---")
            reference_text = self.process_pdf(pdf_path)
            if not reference_text:
                final_result['error_message'] = "Failed to extract any text from the PDF."
                final_result['overall_feedback'] = "Error: Could not extract text from source PDF."
                return final_result

            final_result['source_text_length'] = len(reference_text)
            category_weights = {
                'factual_accuracy': 2.5,
                'completeness': 1.2,
                'chronological_order': 1.0,
                'clinical_relevance': 1.5,
                'organization': 0.8,
            }
            score_keys = list(category_weights.keys()) 
            feedback_keys = [f"{key}_feedback" for key in score_keys] 
            prompt = f"""You are an expert veterinary clinician reviewing a medical history summary.
            Your task is to grade the provided "Summary to Grade" based *only* on the information contained within the "Source Medical Record Full Text".
            Do not use any external knowledge.

            Critically evaluate the summary based on the following criteria: Factual Accuracy, Completeness, Chronological Order, Clinical Relevance, and Organization.
            Provide a score from 1 (Poor) to 5 (Excellent) and specific, constructive feedback for each criterion based *only* on the provided texts.
            Also provide detailed overall feedback

             **Follow these steps:**

             **Step 1: Detailed Analysis (Think thoroughly through each analysis)**
            Before providing the final scores, perform a detailed analysis comparing the Summary to the Source Text for each criterion. Write down your internal thought process for each point:
            1. **Factual Accuracy Analysis:** [Compare specific facts like dates, names, diagnoses, treatments, results in the summary against the source text. Note any matches, mismatches, or fabrications found.]
            2. **Completeness Analysis:** [Identify the key medical events, diagnoses, treatments, and significant findings mentioned in the source text. Check if each key item is present in the summary. Note any significant omissions.]
            3. **Chronological Order Analysis:** [Trace the sequence of events presented in the summary and compare it to the timeline indicated in the source text. Note if the order is correct or incorrect.]
            4. **Clinical Relevance Analysis:** [Assess if the summary focuses on the most medically important information from the source text, appropriate for a referral or history. Note if it includes excessive trivial detail or misses crucial context.]
            5. **Organization Analysis:** [Evaluate the structure, clarity, and flow of the summary. Is it easy to read and logically organized (e.g., chronologically, by problem)? Note strengths or weaknesses in organization.]

            Output your response as a JSON object conforming to the following Pydantic schema:

            ```python
            class Grade(BaseModel):
                factual_accuracy: int = Field(..., description="Score 1-5 for factual accuracy")
                factual_accuracy_feedback: str = Field(..., description="Detailed feedback for factual accuracy")
                completeness: int = Field(..., description="Score 1-5 for completeness")
                completeness_feedback: str = Field(..., description="Detailed feedback for completeness")
                chronological_order: int = Field(..., description="Score 1-5 for chronological order")
                chronological_order_feedback: str = Field(..., description="Detailed feedback for chronological order")
                clinical_relevance: int = Field(..., description="Score 1-5 for clinical relevance")
                clinical_relevance_feedback: str = Field(..., description="Detailed feedback for clinical relevance")
                organization: int = Field(..., description="Score 1-5 for organization")
                organization_feedback: str = Field(..., description="Detailed feedback for organization")
                overall_feedback: str = Field(..., description="Detailed overall feedback text. Be specific")
            ```

            **Source Medical Record Full Text:**
            
            ```
            {reference_text}
            ```
            **Summary to Grade:**
            ```
            {summary}
            ```
            **JSON Output:**"""

            estimated_tokens = len(reference_text) / 3 # Rough estimate
            print(f"Estimated source text tokens (very rough): {estimated_tokens:.0f}")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.grader_config
    )
            parsed_scores = {}
            parsed_feedback = {}
            parsing_errors = []

            try:
                if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
                     raise ValueError("LLM response structure invalid or missing content.")

                content = response.text
                print("Grading response received.")
                parsed_data = json.loads(content)

                for key in score_keys:
                    parsed_scores[key] = parsed_data.get(key)

                    if parsed_scores[key] is None:
                        parsing_errors.append(f"Missing score key in JSON: {key}")
                        print(f"Warning: Missing score key in JSON: {key}")
                    elif not isinstance(parsed_scores[key], int) or not (1 <= parsed_scores[key] <= 5):
                         parsing_errors.append(f"Invalid score value for {key}: {parsed_scores[key]}")
                         print(f"Warning: Invalid score value for {key}: {parsed_scores[key]}")

                for key in feedback_keys:
                    category_key = key.replace('_feedback', '')
                    parsed_feedback[category_key] = parsed_data.get(key)

                    if parsed_feedback[category_key] is None:
                         parsing_errors.append(f"Missing feedback key in JSON: {key}")
                         print(f"Warning: Missing feedback key in JSON: {key}")
                         parsed_feedback[category_key] = "Feedback missing in JSON." 

                final_result['overall_feedback'] = parsed_data.get('overall_feedback')

                if final_result['overall_feedback'] is None:
                    parsing_errors.append("Missing overall_feedback key in JSON")
                    print("Warning: Missing overall_feedback key in JSON")
                    final_result['overall_feedback'] = "Overall feedback missing in JSON."

            except json.JSONDecodeError as e:
                 print(f"ERROR: Failed to parse LLM response as JSON: {e}")
                 print(f"Raw response text: {content}")
                 final_result['error_message'] = "LLM response was not valid JSON."
                 final_result['overall_feedback'] = "Error: LLM response not valid JSON."
                 parsed_scores = {cat: None for cat in score_keys}
                 parsed_feedback = {cat: "Parsing failed" for cat in score_keys}
                 parsing_errors.append("JSONDecodeError")

            except ValueError as e: 
                print(f"ERROR: {e}")
                final_result['error_message'] = str(e)
                final_result['overall_feedback'] = "Error: LLM response structure invalid."
                parsed_scores = {cat: None for cat in score_keys}
                parsed_feedback = {cat: "Parsing failed" for cat in score_keys}
                parsing_errors.append("InvalidResponseStructure")

            except Exception as e: 
                print(f"ERROR: Unexpected error processing LLM response: {e}")
                traceback.print_exc()
                final_result['error_message'] = f"Unexpected error processing response: {e}"
                final_result['overall_feedback'] = "Error: Processing response failed."
                parsed_scores = {cat: None for cat in score_keys}
                parsed_feedback = {cat: "Parsing failed" for cat in score_keys}
                parsing_errors.append(f"UnexpectedProcessingError: {type(e).__name__}")

            valid_scores = {key: score for key, score in parsed_scores.items() if isinstance(score, int) and 1 <= score <= 5}

            if valid_scores:
                weighted_sum = sum(score * category_weights[key] for key, score in valid_scores.items())
                total_weight = sum(category_weights[key] for key in valid_scores.keys())
                weighted_average = weighted_sum / total_weight if total_weight > 0 else None
                final_result['average_score'] = round(weighted_average, 2) if weighted_average is not None else None
            else:
                 final_result['average_score'] = None


            final_result['scores'] = parsed_scores 
            final_result['feedback'] = parsed_feedback
            final_result['pass'] = final_result['average_score'] >= 4.0 if final_result['average_score'] is not None else False

            if parsing_errors:
                 existing_err = final_result.get('error_message')
                 err_str = " | ".join(parsing_errors)
                 final_result['error_message'] = f"{existing_err} | {err_str}" if existing_err else err_str
                 if final_result['overall_feedback'] and not final_result['overall_feedback'].startswith("Error:") and not final_result['overall_feedback'].startswith("Parse Fail"):
                     final_result['overall_feedback'] += f" --- PARSING WARNINGS: {len(parsing_errors)} issue(s) found."

            print(f"Grading complete. Average Score: {final_result['average_score']}. Pass: {final_result['pass']}")
            return final_result

        except FileNotFoundError as e:
             print(f"Error: {e}")
             final_result['error_message'] = str(e)
             final_result['overall_feedback'] = "Error: PDF file not found."
             return final_result
        
        except Exception as e:
            print(f"An unexpected error occurred during grading (Native SDK): {str(e)}")
            traceback.print_exc()
            final_result['error_message'] = f"Unexpected error: {str(e)}"
            final_result.update({
                'scores': {}, 'feedback': {}, 'average_score': None,
                'overall_feedback': "An unexpected error occurred during the grading process.", 'pass': False
            })
            return final_result
        finally:
            reference_text = None
            response = None
            content = None
            gc.collect()

    def create_grading_report(
        grader: PDFSummaryGrader, 
        summaries_path: str,
        pdf_directory: str = "/Users/tylerpoore/Workspace/ani_ml/data/notion_files/extracted/pdfs",
        output_dir: str = '/Users/tylerpoore/Workspace/ani_ml/vet-summary-eval-framework/data/graded'
        ):
        """
        Grades summaries and generates a report including detailed feedback.

        Args:
            grader: An instance of the PDFSummaryGrader class.
            summaries_path: Path to the JSON file containing summaries {pdf_name: summary_text}.
            pdf_directory: Directory containing the source PDF files.
            output_dir: Directory where the report files (Excel, CSV) will be saved.

        Returns:
            pandas.DataFrame: The DataFrame containing the grading results.
        """
        try:
            with open(summaries_path, 'r') as f:
                summaries_dict = json.load(f)
        except FileNotFoundError:
            print(f"Error: Summaries JSON file not found at {summaries_path}")
            return pd.DataFrame()
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {summaries_path}")
            return pd.DataFrame()

        results = []

        graded_file_name = Path(summaries_path).stem

        print(f"Processing {len(summaries_dict)} summaries from {Path(summaries_path).name}...")
        for pdf_name, summary_text in tqdm(summaries_dict.items(), desc="Grading Summaries"):

            if isinstance(summary_text, list) and summary_text:
                summary_text = summary_text[0]
            elif not isinstance(summary_text, str):
                print(f"Warning: Unexpected summary format for {pdf_name}. Skipping.")
                continue
            if not summary_text:
                print(f"Warning: Empty summary provided for {pdf_name}. Skipping.")
                continue

            print(f"\nProcessing {pdf_name}...")

            pdf_full_path = os.path.join(pdf_directory, pdf_name)
            if not os.path.exists(pdf_full_path):
                print(f"Warning: PDF file not found at {pdf_full_path}. Skipping.")
                continue

            try:
                result = grader.grade_summary(pdf_full_path, summary_text)
            except Exception as e:
                print(f"Error during grade_summary call for {pdf_name}: {e}")
                traceback.print_exc() 
                results.append({ 
                    'PDF Name': pdf_name, 'Factual Accuracy': 'Error', 'Completeness': 'Error',
                    'Chronological Order': 'Error', 'Clinical Relevance': 'Error', 'Organization': 'Error',
                    'Average Score': 'Error', 'Pass/Fail': 'Error', 'Overall Feedback': f"Grading failed: {e}",
                    'Factual Accuracy Feedback': 'N/A', 'Completeness Feedback': 'N/A', 'Chronological Order Feedback': 'N/A',
                    'Clinical Relevance Feedback': 'N/A', 'Organization Feedback': 'N/A' 
                })
                continue 

            if result and isinstance(result.get('scores'), dict) and isinstance(result.get('feedback'), dict):
                scores = result['scores']
                feedback = result['feedback']
                row = {
                    'PDF Name': pdf_name,
                    'Factual Accuracy': scores.get('factual_accuracy'),
                    'Completeness': scores.get('completeness'),
                    'Chronological Order': scores.get('chronological_order'),
                    'Clinical Relevance': scores.get('clinical_relevance'),
                    'Organization': scores.get('organization'),
                    'Factual Accuracy Feedback': feedback.get('factual_accuracy', 'N/A'),
                    'Completeness Feedback': feedback.get('completeness', 'N/A'),
                    'Chronological Order Feedback': feedback.get('chronological_order', 'N/A'),
                    'Clinical Relevance Feedback': feedback.get('clinical_relevance', 'N/A'),
                    'Organization Feedback': feedback.get('organization', 'N/A'),
                    'Average Score': result.get('average_score'),
                    'Pass/Fail': 'Pass' if result.get('pass') else 'Fail',
                    'Overall Feedback': result.get('overall_feedback', 'N/A')                }
                results.append(row)
            else:
                print(f"Warning: Could not parse scores/feedback properly for {pdf_name}. Feedback: {result.get('overall_feedback', 'N/A')}")
                results.append({ 
                    'PDF Name': pdf_name, 'Factual Accuracy': 'Parse Fail', 'Completeness': 'Parse Fail',
                    'Chronological Order': 'Parse Fail', 'Clinical Relevance': 'Parse Fail', 'Organization': 'Parse Fail',
                    'Average Score': result.get('average_score', 'Parse Fail'), 'Pass/Fail': 'Fail',
                    'Overall Feedback': result.get('overall_feedback', 'Score/Feedback parsing failed.'),
                    'Factual Accuracy Feedback': result.get('feedback', {}).get('factual_accuracy', 'Parse Fail'),
                    'Completeness Feedback': result.get('feedback', {}).get('completeness', 'Parse Fail'),
                    'Chronological Order Feedback': result.get('feedback', {}).get('chronological_order', 'Parse Fail'),
                    'Clinical Relevance Feedback': result.get('feedback', {}).get('clinical_relevance', 'Parse Fail'),
                    'Organization Feedback': result.get('feedback', {}).get('organization', 'Parse Fail'),
                })

        if not results:
            print("No summaries were successfully processed or parsed.")
            return pd.DataFrame() 

        df = pd.DataFrame(results)

        score_columns = ['factual_accuracy', 'completeness', 'chronological_order',
                        'clinical_relevance', 'organization', 'average_score']
        df.columns = [col.replace(' ', '_').lower() if isinstance(col, str) else col for col in df.columns] 
        for col in score_columns:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce')

        column_rename_map = {
            'pdf_name': 'PDF Name',
            'factual_accuracy': 'Factual Accuracy',
            'completeness': 'Completeness',
            'chronological_order': 'Chronological Order',
            'clinical_relevance': 'Clinical Relevance',
            'organization': 'Organization',
            'average_score': 'Average Score',
            'pass_fail': 'Pass/Fail',
            'overall_feedback': 'Overall Feedback',
            'factual_accuracy_feedback': 'Factual Accuracy Feedback',
            'completeness_feedback': 'Completeness Feedback',
            'chronological_order_feedback': 'Chronological Order Feedback',
            'clinical_relevance_feedback': 'Clinical Relevance Feedback',
            'organization_feedback': 'Organization Feedback'
        }
        df = df.rename(columns=column_rename_map)

        ordered_columns = [
            'PDF Name',
            'Factual Accuracy', 'Factual Accuracy Feedback',
            'Completeness', 'Completeness Feedback',
            'Chronological Order', 'Chronological Order Feedback',
            'Clinical Relevance', 'Clinical Relevance Feedback',
            'Organization', 'Organization Feedback',
            'Average Score', 'Pass/Fail', 'Overall Feedback'
        ]
        ordered_columns = [col for col in ordered_columns if col in df.columns]
        df = df[ordered_columns]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True) 
        base_filename = f'{graded_file_name}_grading_report_{timestamp}' 
        excel_path = os.path.join(output_dir, f'{base_filename}.xlsx')
        csv_path = os.path.join(output_dir, f'{base_filename}.csv')

        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Summary Scores', index=False)
                worksheet = writer.sheets['Summary Scores']
                for idx, col_name in enumerate(df.columns):
                    series = df[col_name]
                    max_len = max((
                        series.astype(str).map(len).max(),
                        len(str(col_name))
                        )) + 2 
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 80) 
            print(f"\nResults saved to Excel: {excel_path}")
        except Exception as e:
            print(f"\nError saving Excel file: {e}")

        try:
            df.to_csv(csv_path, index=False)
            print(f"Results saved to CSV: {csv_path}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

        return df