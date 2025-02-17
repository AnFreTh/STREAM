from openai import OpenAI, OpenAIError
from loguru import logger
import os


# Define allowed models
ALLOWED_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
]


def story_topic(
    topic,
    api_key,
    model="gpt-3.5-turbo-16k",
    content="You are a creative writer.",
    prompt="Create a creative story that includes the following words:",
    max_tokens=250,
    temperature=0.8,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
):
    """
    Generate a creative story using OpenAI's GPT model.

    Parameters:
    - topic: List of words or phrases to include in the story.
    - api_key: API key for OpenAI.
    - model: Model to use (e.g., 'gpt-3.5-turbo-16k').
    - content: Initial system content.
    - prompt: Prompt for the story generation.
    - max_tokens: Maximum tokens for the response.
    - temperature: Creativity level for the model.
    - top_p: Nucleus sampling parameter.
    - frequency_penalty: Penalty for word frequency.
    - presence_penalty: Penalty for word presence.

    Returns:
    - story: Generated story as a string.
    """

    # Load the API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the OpenAI client with your API key
    client = OpenAI(api_key=api_key)

    # Validate model
    if model not in ALLOWED_MODELS:
        raise ValueError(
            f"Invalid model. Please choose a valid model from {ALLOWED_MODELS}."
        )

    # Create the prompt
    prompt = f"{prompt}: {', '.join(topic)}. Make it as short as {max_tokens} words."

    # Logging the operation
    logger.info(f"--- Generating story with model: {model} ---")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # Ensure the response is valid
        if response and len(response.choices) > 0:
            story = response.choices[0].message.content
        else:
            story = "No story generated. Please try again."

        return story

    except OpenAIError as e:
        return f"An error occurred: {str(e)}"
