from openai import OpenAI, OpenAIError
from loguru import logger
import os
from IPython.display import Image, display

# Define allowed models
ALLOWED_MODELS = ["dall-e-3", "dall-e-2"]


def movie_poster(
    topic,
    api_key,
    model="dall-e-3",
    quality="standard",
    prompt="Create a movie poster that depicts that topic best. Note that the words are ordered in decreasing order of their importance.",
    size="1024x1024",
    return_style="url",
):
    """
    Generate a movie-poster-style image based on a given topic using OpenAI's DALL-E model.

    Parameters:
    - topic: List of words/phrases or list of tuples (word, importance) representing the topic.
    - api_key: API key for OpenAI.
    - model: Model to use (e.g., 'dall-e').
    - poster_style: Description of the style for the image, default is "Movie Poster".
    - content: Initial system content.
    - prompt: Prompt for the image generation.
    - size: Size of the generated image, default is "1024x1024".

    Returns:
    - image_url: URL of the generated image.
    """

    # Load the API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("API key is missing. Please provide an API key.")

    assert return_style in [
        "url",
        "plot",
    ], "Invalid return style. Please choose 'url' or 'plot'"

    # Initialize the OpenAI client with your API key
    client = OpenAI(api_key=api_key)

    # Validate model
    if model not in ALLOWED_MODELS:
        raise ValueError(
            f"Invalid model. Please choose a valid model from {ALLOWED_MODELS}."
        )

    # Create the prompt for the movie poster
    if isinstance(topic[0], tuple):
        # If the topic is a list of tuples with importance
        topic_description = ", ".join(
            [f"{word} (importance: {importance})" for word, importance in topic]
        )
    else:
        # If the topic is a list of words in descending importance
        topic_description = topic

    image_prompt = f"Given the following topic: {topic_description}. {prompt}"

    # Logging the operation
    logger.info(f"--- Generating image with model: {model} ---")
    response = client.images.generate(
        model=model,
        prompt=image_prompt,
        size=size,
        quality=quality,
        n=1,
    )

    # Ensure the response is valid
    if response:
        image_url = response.data[0].url
    else:
        image_url = "No image generated. Please try again."

    if return_style == "url":
        return image_url

    elif return_style == "plot":
        display(Image(url=image_url))
