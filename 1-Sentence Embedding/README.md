## Tokenizer Overview
The `tokenizer` is a tool that converts raw text into a format suitable for machine learning models, typically by splitting the text into smaller units called **tokens**. These tokens can be words, subwords, or characters, depending on the tokenizer's configuration. The tokenizer also maps these tokens to their corresponding numerical IDs, which are used as input for models.

### Key Features of tokenization:
1. **Tokenization**: Splits text into tokens.
2. **Encoding**: Converts tokens into numerical IDs.
3. **Special Tokens**: Adds special tokens (e.g., `[CLS]`, `[SEP]`) if required by the model.
4. **Padding and Truncation**: Ensures all sequences have the same length for batch processing.

---

## Parameters Explained

### `return_tensors="pt"`
- Specifies the format of the returned output.
- `"pt"` stands for PyTorch, meaning the output will be in the form of PyTorch tensors.


### `padding=True`
- **Padding** ensures that all sequences in a batch have the same length by adding special padding tokens (usually `[PAD]`) to shorter sequences.
- When `padding=True`, the tokenizer automatically pads sequences to the length of the longest sequence in the batch.

### `truncation=True`
- **Truncation** ensures that sequences longer than a specified maximum length are cut off (truncated) to fit.
- This is important because models often have a maximum input length (e.g., 512 tokens for BERT).

---

## Example Workflow
1. **Input Text**: `["Hello, world!", "How are you?"]`
2. **Tokenization**: Splits text into tokens (e.g., `["Hello", ",", "world", "!"]`).
3. **Encoding**: Converts tokens to IDs (e.g., `[101, 7592, 1010, 2088, 999, 102]`).
4. **Padding**: Adds padding tokens to shorter sequences to match the longest sequence.
5. **Truncation**: Cuts off tokens from sequences longer than the model's max length.
6. **Output**: Returns a dictionary of PyTorch tensors (e.g., `{"input_ids": tensor(...), "attention_mask": tensor(...)}`).

---

# Understanding Embeddings

Embeddings are a powerful way to represent words or tokens in a numerical format that captures their meaning and relationships. Here are two intuitive ways to think about embeddings:

## 1. Embeddings as Coordinates on a Map
Imagine a map where cities like NYC and Los Angeles are marked. If you only know their names, you can't tell their locations or how far apart they are. However, if you are given their coordinates (e.g., latitude and longitude), you can see that they are on opposite sides of the same country. 

Similarly, embeddings assign "coordinates" to each token (word or subword) in a high-dimensional space. These coordinates help the model understand how close or far apart the meanings of tokens are. For example, the embeddings for "cat" and "dog" will be closer to each other than to the embedding for "car," reflecting their semantic similarity.

## 2. Embeddings as Meaning Profiles
An embedding vector can also be thought of as a "meaning profile" for a word. Each number in the vector represents how much the word "scores" on a particular aspect of meaning. For instance, consider a 3-dimensional embedding vector for the word "dog":

```
[0.8, 0.2, 0.1]
```

- `0.8` means "dog" scores high on the "animal" aspect.
- `0.2` means "dog" scores low on the "food" aspect.
- `0.1` means "dog" scores very low on the "action" aspect.

In this way, embeddings capture the nuanced meaning of words in a structured, numerical format, making it easier for models to process and understand language.

---

By using embeddings, language models can effectively represent and reason about the relationships and meanings of words, enabling more accurate and context-aware predictions.
