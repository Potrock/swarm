from swarm import Swarm, Agent
from swarm.types import Result
from PIL import Image

client = Swarm()


def view_screenshot() -> str:
    base64_img = "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"
    import base64
    import io
    
    # Convert base64 to PIL Image
    image_bytes = base64.b64decode(base64_img)
    image = Image.open(io.BytesIO(image_bytes))
    
    return Result(
        value="Here is the screenshot.",
        image=image
    )


agent = Agent(
    name="Agent",
    instructions="You are a helpful agent. You can view the user's screen with the view_screenshot tool",
    functions=[view_screenshot],
)

messages = [{"role": "user", "content": "What's on my screen?"}]

response = client.run(agent=agent, messages=messages, debug=True)
print(response.messages[-1]["content"])