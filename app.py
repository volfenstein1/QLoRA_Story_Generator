import streamlit as st
import textstat
import torch
from transformers.generation.streamers import TextIteratorStreamer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from threading import Thread

st.set_page_config(page_title="Simple Story Generator")


@st.cache_resource
def load_resources():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "volfenstein/LORA-simple-stories-generator-v2"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "volfenstein/LORA-simple-stories-generator-v2"
    )
    return tokenizer, model


tokenizer, model = load_resources()


def generate_story_stream(topic, theme, wordcount, complexity):
    user_prompt = """Topic: {topic}

    Theme: {theme}

    Wordcount: {wordcount}

    Complexity: {complexity}""".format(
        topic=topic,
        theme=theme,
        wordcount=wordcount,
        complexity=complexity,
    )

    tokenized_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )

    # outputs = model.generate(
    #     tokenized_chat,
    #     max_new_tokens=512,
    #     pad_token_id=tokenizer.eos_token_id,
    #     streamer=streamer,
    # )

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=False
    )

    thread = Thread(
        target=model.generate,
        kwargs=dict(
            **tokenized_chat,
            max_new_tokens=512,
            streamer=streamer,
        ),
    )
    thread.start()

    full_story = ""
    for new_text in streamer:
        full_story += new_text
        yield new_text

    thread.join()

    # story = tokenizer.decode(outputs[0])
    # return story
    # inputs = tokenizer([messages], return_tensors="pt").to(model.device)
    # streamer = TextIteratorStreamer(
    #     tokenizer, skip_prompt=True, skip_special_tokens=True
    # )
    # thread = Thread(
    #     target=model.generate,
    #     kwargs=dict(
    #         **inputs,
    #         max_new_tokens=512,
    #         streamer=streamer,
    #     ),
    # )
    # thread.start()

    # full_story = ""
    # for new_text in streamer:
    #     full_story += new_text
    #     yield new_text

    # thread.join()


def generate_story_stream2(topic, theme, wordcount, complexity):
    user_prompt = """Topic: {topic}

    Theme: {theme}

    Wordcount: {wordcount}

    Complexity: {complexity}""".format(
        topic=topic,
        theme=theme,
        wordcount=wordcount,
        complexity=complexity,
    )

    messages = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Tokenize the messages string
    model_inputs = tokenizer([messages], return_tensors="pt").to(model.device)
    # Remove token_type_ids if present, as some models do not use them
    if "token_type_ids" in model_inputs:
        del model_inputs["token_type_ids"]
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=False
    )
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        repetition_penalty=1.5,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Initialize an empty string to store the generated text
    global story
    for new_text in streamer:
        if "EOS" in new_text:
            # response += new_text.split('[EOS]')[0]
            return story
        story += new_text
        yield new_text


def word_count(text):
    punctuations = ".,!?;:'\"()-"
    for p in punctuations:
        text = text.replace(p, " ")
    # Split into words by whitespace
    words = text.split()
    return len(words)


themes = [
    "Family",
    "Deception",
    "Consciousness",
    "Growth",
    "Transformation",
    "Problem-Solving",
    "Magic",
    "Dreams",
    "Discovery",
    "Morality",
    "Coming of age",
    "Belonging",
    "Logic",
    "Celebration",
    "Planning",
    "Overcoming",
    "Friendship",
    "Honesty",
    "Helping Others",
    "Hardship",
    "The Five Senses",
    "Independence",
    "Amnesia",
    "Surprises",
    "Conscience",
    "Imagination",
    "Failure",
    "Agency",
    "Self-Acceptance",
    "Courage",
    "Hope",
    "Cooperation",
    "Humor",
    "Power",
    "Adventure",
    "Kindness",
    "Loss",
    "Strategy",
    "Curiosity",
    "Conflict",
    "Revenge",
    "Generosity",
    "Perseverance",
    "Scheming",
    "Travel",
    "Resilience",
    "Resourcefulness",
    "Teamwork",
    "Optimism",
    "Love",
]

topics = [
    "fantasy worlds",
    "hidden treasures",
    "magical objects",
    "royal kingdoms",
    "fairy tales",
    "the arts",
    "talking animals",
    "dream worlds",
    "riddles",
    "cultural traditions",
    "alien encounters",
    "subterranean worlds",
    "lost civilizations",
    "magical lands",
    "sports",
    "time travel",
    "haunted places",
    "gardens",
    "mystical creatures",
    "virtual worlds",
    "mysterious maps",
    "island adventures",
    "undercover missions",
    "unusual vehicles",
    "shape-shifting",
    "the sky",
    "school life",
    "invisibility",
    "robots and technology",
    "seasonal changes",
    "space exploration",
    "holidays",
    "sibling rivalry",
    "secret societies",
    "treasure hunts",
    "dinosaurs",
    "snowy adventures",
    "giant creatures",
    "a deadline or time limit",
    "pirates",
    "superheroes",
    "bygone eras",
    "outer space",
    "living objects",
    "lost cities",
    "enchanted forests",
    "underwater adventures",
    "miniature worlds",
]

left, right = st.columns(2, vertical_alignment="bottom")

selected_theme = left.selectbox(
    "Theme",
    themes,
    index=None,
    placeholder="Select a theme...",
)

selected_topic = right.selectbox(
    "Topic",
    topics,
    index=None,
    placeholder="Select a topic...",
)

selected_wordcount = st.slider("Target word count:", 100, 500, step=25)

selected_complexity = st.slider("Complexity:", 1, 6)

submit = st.button("Generate")

if selected_theme and selected_topic and submit:
    story = ""
    stream = generate_story_stream2(
        topic=selected_topic,
        theme=selected_theme,
        wordcount=selected_wordcount,
        complexity=selected_complexity,
    )
    st.write_stream(stream)

    st.write(
        "Word count:",
        word_count(story),
        "Flesch Kincaid Grade:",
        round(textstat.flesch_kincaid_grade(story), 2),
    )
