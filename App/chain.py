import os 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv


load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-70b-versatile",temperature=0,groq_api_key = os.getenv("GROQ_API_KEY"))

    def extract_jobs(self,cleaned_text):
        prompt_extract = PromptTemplate.from_template(
        """
            ### SCRAPTED TEXT FROM WEBSITE 
            {page_data}
            ### INSTRUCTION 
            The scrapped data is from the careers page of a website. Your job is to extract the job postings and return them in valid JSON format containing the following keys: 'role', 'experience', 'skills', 'description'.
            ### OUTPUT 
            Return only the JSON object directly, with no backticks, escape characters, or any additional text. Ensure it is properly formatted as JSON.
            """
        )

        chain_extract = prompt_extract | self.llm 
        res = chain_extract.invoke(input = {'page_data': cleaned_text })
        try:
            json_parser = JsonOutputParser()
            result = json_parser.parse(res.content)
        except OutputParserException as e:
            raise e("Context too big. Unable to parse jobs.") 
        return result if isinstance(result,list) else [result]
    
    def write_email(self, job, links):
        prompt_email = PromptTemplate.from_template(
        """
            ### JOB DESCRIPTION
            {Job_description}
            ### INSTRUCTION 
            you are mohan, a business development executive at NoVA. NoVA is an AI and software consultingcompany
            the seamless integeration of business process through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solution, process 
            optimization, cost reduction and heightened overall efficiency ,
            Your job is to write a cold email to the client regarding the job mentioned above and fulfilling their needs.
            Remember you are Mohan, BDE at NoVA
            DO not provide PREAMBLE
            ### EMAIL (No preamble) 
            """
        )

        email_extract = prompt_email | self.llm 
        res2 = email_extract.invoke(input = {'Job_description': str(job) })
        return res2.content

    if __name__ == "__main__":
        print(os.getenv("GROQ_API_KEY"))