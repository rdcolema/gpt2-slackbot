from gpt2_bot import generate_samples


def trump(command):
    """Uses GPT-2 model trained on Trump speeches to generate text sample"""
    command = command.split("trump")[1]
    return generate_samples("trump", sample_len=100, prime_text=command)


def okcupid(command):
    """Uses GPT-2 model trained on okcupid dating profiles to generate text sample"""
    command = command.split("okcupid")[1]
    return generate_samples("okcupid", sample_len=100, prime_text=command)


def rap(command):
    """Uses GPT-2 model trained on rap lyrics to generate text sample"""
    command = command.split("rap")[1]
    return generate_samples("rap", prime_text=command)


def ruby(command):
    """Uses GPT-2 model trained on ruby source code to generate text sample"""
    command = command.split("ruby")[1]
    return generate_samples("ruby", prime_text=command)
