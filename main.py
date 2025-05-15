import json
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from mytextsplitters import CustomTextSplitter
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class LLmResponse(BaseModel):
    question: List[str]
    answer: List[str]


class QADatasetGenerator:
    def __init__(self, output_file="hack.json"):
        self.output_file = output_file
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        self.text_splitter = CustomTextSplitter()
        self.data = self._load_existing_data()

    def _load_existing_data(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {"question": [], "answer": []}

    def _save_data(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def get_llm_response(self, context):
        try:
            template = """
You are an AI language expert tasked with generating high-quality question-answer pairs for training a fine-tuned language model.

Given the following context, produce **10 question and answer pairs**. Each question should be clear, specific, and relevant to the context. Each answer should be a **complete, informative sentence** that provides meaningful insight—not just a short or one-word response.

**Guidelines:**
- Focus on factual and context-rich questions.
- Ensure each answer is grammatically correct and elaborates on the topic.
- Avoid repetition across questions.
- Keep the language natural and human-like.

**Output Format:**
Return a valid JSON object with two keys:
- "question": a list of 10 questions.
- "answer": a list of 10 corresponding answers.

Make sure the lists are aligned (i.e., each question has a matching answer at the same index).

**Context:**
{context}
"""
            prompt = PromptTemplate.from_template(template)
            chain = (
                {"context": RunnablePassthrough()}
                | prompt
                | self.llm.with_structured_output(LLmResponse)
            )
            return chain.invoke(context)
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return None

    def process_and_save_incrementally(self, file_path):
        documents = self.text_splitter.load_text(file_path)
        chunks = self.text_splitter.split_text(documents)

        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}/{len(chunks)}...")
            response = self.get_llm_response(chunk.page_content)

            if response:
                self.data["question"].extend(response.question)
                self.data["answer"].extend(response.answer)
                self._save_data()
                print(f"Saved {len(response.question)} Q&A pairs to '{self.output_file}'")
            else:
                print(f"Skipping chunk {i+1} due to an error.")


if __name__ == "__main__":
    file_path = "3.pdf"
    generator = QADatasetGenerator()
    generator.process_and_save_incrementally(file_path)
    
