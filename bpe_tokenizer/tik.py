import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
text = "I spent a spectacular summer."
tokens = enc.encode(text)
decoded = enc.decode(tokens)

print(tokens)
for token in tokens:
    print(token, enc.decode_single_token_bytes(token))
print(decoded)
