import os
import socket
import vertexai
from src.utils.paths import ROOT_FOLDER


class KeyHandler:
    openai_key = 'open_ai_key'
    claude_key = 'claude_key'
    hf_key = 'hf_token'
    google_project_id = 'google_project_id'
    google_project_location = 'google_project_location'

    @classmethod
    def set_env_key(cls):
        os.environ['OPENAI_API_KEY'] = cls.openai_key
        os.environ['ANTHROPIC_API_KEY'] = cls.claude_key
        os.environ['HF_API_KEY'] = cls.hf_key

        # TODO - make sure you have the google service key in the root folder of the project to use Gemini.
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(ROOT_FOLDER / 'google_service_key.json')

        vertexai.init(project=cls.google_project_id, location=cls.google_project_location)

