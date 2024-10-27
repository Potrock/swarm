from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import List, Callable, Union, Optional, Dict, Any
import base64
from io import BytesIO
from PIL import Image

# Third-party imports
from pydantic import BaseModel, model_validator

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.
    
    Attributes:
        value (str): The result value as a string.
        image (Optional[Union[str, Image.Image]]): Optional image data, can be base64 string or PIL Image.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    model_config = {"arbitrary_types_allowed": True}

    value: str = ""
    image: Optional[Union[str, Image.Image]] = None 
    agent: Optional[Agent] = None
    context_variables: dict = {}

    @model_validator(mode="after")
    def validate_and_convert_image(self) -> "Result":
        if self.image is None:
            return self
            
        if isinstance(self.image, Image.Image):
            buffered = BytesIO()
            self.image.save(buffered, format="PNG")
            self.image = base64.b64encode(buffered.getvalue()).decode()
        elif not isinstance(self.image, str):
            raise ValueError("Image must be either a base64 string or PIL Image")
            
        return self
