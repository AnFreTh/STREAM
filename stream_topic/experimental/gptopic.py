from openai import OpenAI, OpenAIError
from loguru import logger
import os
from IPython.display import Image, display

# Define allowed models
ALLOWED_MODELS_POSTER = ["dall-e-3", "dall-e-2"]


ALLOWED_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
]


def topic_summaries(
    topics,
    api_key,
    model="gpt-3.5-turbo-16k",
    content="You are a creative writer.",
    prompt="Provide a 1-2 sentence summary for the following topic:",
    max_tokens=60,
    temperature=0.7,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
):
    """
    Generate a 1-2 sentence summary for each topic using OpenAI's GPT model.

    Parameters:
    - topics: List of lists, where each sublist contains words/phrases representing a topic.
    - api_key: API key for OpenAI.
    - model: Model to use (e.g., 'gpt-3.5-turbo-16k').
    - content: Initial system content.
    - prompt: Prompt for the summary generation.
    - max_tokens: Maximum tokens for each summary.
    - temperature: Creativity level for the model.
    - top_p: Nucleus sampling parameter.
    - frequency_penalty: Penalty for word frequency.
    - presence_penalty: Penalty for word presence.

    Returns:
    - summaries: List of summaries corresponding to each topic.
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

    summaries = []

    for idx, topic in enumerate(topics):
        # Create the prompt for each topic
        topic_prompt = f"{prompt} {', '.join(topic)}."

        # Logging the operation
        logger.info(f"--- Generating summary for topic {idx} with model: {model} ---")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": content},
                    {"role": "user", "content": topic_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            # Ensure the response is valid
            if response and len(response.choices) > 0:
                summary = response.choices[0].message.content
            else:
                summary = "No summary generated. Please try again."

            summaries.append(summary)

        except OpenAIError as e:
            summaries.append(f"An error occurred: {str(e)}")

    return summaries


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
