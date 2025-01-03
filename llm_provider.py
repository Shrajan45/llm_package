import openai
import pandas as pd
from transformers import pipeline
from .prompt_handler import PromptHandler


class LLMProvider():
    def __init__(self,provider:str,api_key:str,prompt_template:str,max_tokens=4096,hf_model=None):
        self.provider=provider
        self.api_key=api_key
        self.max_tokens=max_tokens
        self.hf_model=hf_model
        self.prompt_handler=PromptHandler(prompt_template)

        if self.provider=="huggingface" and self.hf_model:
             self.hf_pipline=pipeline('text-generation',model=hf_model)
        
    def querry(self,prompt:str):
        if self.provider=='openai':
            openai.api=self.api_key
            response = openai.Completion.create(engine="text-davinci-003",
                                                prompt=prompt,
                                                max_tokens=self.max_tokens)
            return response.choices[0].text.strip()
        else:
            raise ValueError(f"unsupported provider {self.provider}")


    def _convert_to_dicts(self,rows):
        if isinstance(rows,pd.DataFrame):
            return rows.to_dict(orient='records')
    
    def process_row_by_row(self, rows):
            """
            Process input row-by-row. This method generates a prompt for each row 
            and queries the LLM for a response.
            """
            rows = self._convert_to_dicts(rows)
            results = []
            for row in rows:
                prompt = self.prompt_handler.generate_prompt(**row)  # Use prompt_handler to generate the prompt
                results.append(self.query(prompt))  # Query with the generated prompt
            return results
    
    def process_in_one_big_chunk(self,rows):            
            rows=self._convert_to_dicts(rows)
            formated_rows=[self.prompt_handler.generate_prompt(**row) for row in rows]
            prompts="\n".join(formated_rows)
            if self.count_tokens(prompts)>self.max_tokens:
                 raise ValueError("Total prompt exceeds the maximum token limit")            
            return self.querry(prompts)
    
    def count_tokens(self,text:str):
         return len(text.split())