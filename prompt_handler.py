class PromptHandler:

    def __init__(self,prompt_template:str):
        self.prompt_template=prompt_template

    def generate_prompt(self,**kwargs):
        return self.prompt_template.format(**kwargs)
    