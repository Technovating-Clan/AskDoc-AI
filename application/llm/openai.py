from application.llm.base import BaseLLM
from application.core.settings import settings

import os
import openai
import requests

class OpenAILLM(BaseLLM):

    def __init__(self, api_key=None, user_api_key=None, *args, **kwargs):
        from openai import OpenAI

        super().__init__(*args, **kwargs)
        if settings.OPENAI_BASE_URL:
            self.client = OpenAI(
                api_key=api_key,
                base_url=settings.OPENAI_BASE_URL
            )
        else:
            self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.user_api_key = user_api_key

    def _raw_gen(
        self,
        baseself,
        model,
        messages,
        stream=False,
        engine=settings.AZURE_DEPLOYMENT_NAME,
        **kwargs
    ):  
        response = self.client.chat.completions.create(
            model=model, messages=messages, stream=stream, **kwargs
        )

        return response.choices[0].message.content

    def _raw_gen_stream(
        self,
        baseself,
        model,
        messages,
        stream=True,
        engine=settings.AZURE_DEPLOYMENT_NAME,
        **kwargs
    ):  
        response = self.client.chat.completions.create(
            model=model, messages=messages, stream=stream, **kwargs
        )

        for line in response:
            # import sys
            # print(line.choices[0].delta.content, file=sys.stderr)
            if line.choices[0].delta.content is not None:
                yield line.choices[0].delta.content

class OpenAIProvider:
    
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key as an environment variable

    def generate_questions(self, content):
        prompt = f"Generate 5 insightful questions based on the following content:\n\n{content}\n\nQuestions:"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or any model you want to use
            messages=[{"role": "user", "content": prompt}]
        )
        
        questions = response.choices[0].message['content'].strip().split('\n')
        return questions


class AzureOpenAILLM(OpenAILLM):

    def __init__(
        self, openai_api_key, openai_api_base, openai_api_version, deployment_name
    ):
        super().__init__(openai_api_key)
        self.api_base = (settings.OPENAI_API_BASE,)
        self.api_version = (settings.OPENAI_API_VERSION,)
        self.deployment_name = (settings.AZURE_DEPLOYMENT_NAME,)
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=openai_api_key,
            api_version=settings.OPENAI_API_VERSION,
            api_base=settings.OPENAI_API_BASE,
            deployment_name=settings.AZURE_DEPLOYMENT_NAME,
        )

