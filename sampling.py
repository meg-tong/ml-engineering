# %%
from typing import List, Tuple

import numpy as np
import torch as t
import transformers

# %%
if __name__ == "__main__":
    gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# %%
def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)

def get_logits(new_input_ids, tokenizer, model, device, *args):
    new_input_ids_truncated = new_input_ids[-min(tokenizer.model_max_length, new_input_ids.shape[0]):].unsqueeze(0)
    output = model(new_input_ids_truncated, *args)
    all_logits = output if isinstance(output, t.Tensor) else output.logits
    return all_logits[0, -1] #batch=0, seq_len=-1 -> returns vocab_size

def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(np.array(input_ids + generated), dtype=t.int64, device=device)
        logits = get_logits(new_input_ids, tokenizer, model, device)
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)

# %%
def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    return logits.argmax().numpy()

if __name__ == "__main__":
    prompt = "Jingle bells, jingle bells, jingle all the way"
    print("Greedy decoding with prompt: ", prompt)
    output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
    print(f"Your model said: {output}")
    expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
    assert output == expected

    print("Greedy decoding a second time (should be deterministic): ")
    output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
    print(f"Your model said: {output}")
    expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
    assert output == expected

    print("Tests passed!")
# %%
def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    return t.distributions.categorical.Categorical(logits=logits).sample()

if __name__ == "__main__":
    N = 20000
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    t.testing.assert_close(counts, probs, atol=0.01, rtol=0)
    print("Tests passed!")
# %%
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    return logits / temperature

if __name__ == "__main__":
    logits = t.tensor([1, 2]).log()
    cold_logits = apply_temperature(logits, 0.001)
    print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
    t.testing.assert_close(cold_logits, 1000.0 * logits)
    hot_logits = apply_temperature(logits, 1000.0)
    print("A high temperature flattens the distribution: ", hot_logits)
    t.testing.assert_close(hot_logits, 0.001 * logits)
    print("Tests passed!")

# %%
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    count = input_ids.bincount(minlength=len(logits))
    logits -= count * freq_penalty
    return logits

if __name__ == "__main__":
    bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
    input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt").squeeze()
    logits = t.ones(tokenizer.vocab_size)
    penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
    assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
    assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
    print("Tests passed!")
# %%
N_RUNS = 1
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(freq_penalty=100.0)),
    ("Negative freq penalty", dict(freq_penalty=-1.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]
if __name__ == "__main__":
    for (name, kwargs) in cases:
        for i in range(N_RUNS):
            output = sample_tokens(gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
            print(f"Sample {i} with: {name} ({kwargs}):")
            print(f"Your model said: {repr(output)}\n")
# %%
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    values, indices = t.topk(logits, top_k)
    return indices[sample_basic(values)].item()

if __name__ == "__main__":
    N = 50000
    k = 3
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[:-k] = 0
    expected /= expected.sum()
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    t.testing.assert_close(counts, expected, atol=0.01, rtol=0)
    print("Tests passed!")
# %%
if __name__ == "__main__":
    your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
    print(f"Your model said: {repr(output)}")
# %%
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    probs = t.exp(logits.double()) / t.exp(logits.double()).sum()
    sorted_probs, sorted_indices = probs.sort(descending=True)
    cum_probs = sorted_probs.cumsum(-1)
    last_index = max(min_tokens_to_keep, t.where(cum_probs >= top_p)[0][0].numpy() + 1)
    masked_probs = sorted_probs[:last_index]
    sample = t.distributions.categorical.Categorical(probs=t.tensor(masked_probs)).sample()
    return sorted_indices[sample]

if __name__ == "__main__":
    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print("top_p of 0.5 or lower should only return token 2: ", counts)
    assert counts[0] == 0 and counts[1] == 0

    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
    assert counts[0] == 0

    N = 50000
    top_p = 0.71
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[0:2] = 0
    expected /= expected.sum()
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    t.testing.assert_close(counts, expected, atol=0.01, rtol=0.0)

    print("All tests passed!")
# %%
if __name__ == "__main__":
    your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
    output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
    print(f"Your model said: {repr(output)}")
# %%
def beam_search(
    model, input_ids: t.Tensor, num_return_sequences: int, num_beams: int, max_new_tokens: int, tokenizer, verbose=False
) -> List[Tuple[float, t.Tensor]]:
    '''
    input_ids: (seq, ) - the prompt
    max_new_tokens: stop after this many new tokens are generated, even if no EOS is generated. In this case, the best incomplete sequences should also be returned.
    verbose: if True, print the current (unfinished) completions after each iteration for debugging purposes

    Return list of length num_return_sequences. Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, sorted by descending logprob.
    '''
    assert num_return_sequences <= num_beams
    model.eval()
    device = next(model.parameters()).device
    to_be_continued = [(0, input_ids)]
    finished = []
    with t.inference_mode():
        for _ in range(max_new_tokens):
            print("Adding {} tokens".format(_))
            possibilities = []
            for completion_log_prob, completion in to_be_continued:
                new_input_ids = t.tensor(np.array(completion), dtype=t.int64, device=device)
                logits = get_logits(new_input_ids, tokenizer, model, device)
                log_probs = t.nn.functional.log_softmax(logits, dim=0)
                k_log_probs, k_indices = t.topk(log_probs, num_beams)
                for log_prob, index in zip(k_log_probs, k_indices):
                    possibility = (completion_log_prob + log_prob, list(completion) + [index.item()])
                    if index == getattr(tokenizer, "eos_token_id", None):
                        finished.append(possibility)
                    possibilities.append(possibility)
            to_be_continued = sorted(possibilities, key=lambda x: x[0], reverse=True)[:num_beams]
            if verbose:
                for log_prob, completion in to_be_continued:
                    print(log_prob, tokenizer.decode(completion))
                print()
    finished = finished + to_be_continued
    return sorted(finished, key=lambda x: x[0], reverse=True)[:num_return_sequences]
    #

if __name__ == "__main__":
    #tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    #gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").to(device).train()

    your_prompt = "I don't want to rule the universe. I just think"
    input_ids = tokenizer(your_prompt, return_tensors="pt", return_attention_mask=False)["input_ids"][0]

    num_return_sequences = 3
    num_beams = 6
    max_new_tokens = 10

    final_logitsums_and_completions = beam_search(gpt, input_ids, num_return_sequences, num_beams, max_new_tokens, tokenizer, verbose=True)
    for log_prob, completion in final_logitsums_and_completions:
        print(log_prob, tokenizer.decode(completion))
# %%
