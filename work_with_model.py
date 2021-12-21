import pickle


def check_review(text, model, ft):
    for char in ['\n', '\t', '.', ',', '!', '?', '_', '(', ')', ':', ';', '/', '<', '>', '^', '"', '&']:
        text = text.replace(char, ' ')
    for i in range(10):
        text = text.replace(str(i), ' ')
    text = text.strip(' ').rstrip(' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    res = model.predict_proba([ft.get_sentence_vector(text)])[:, 1][0]
    if res < 0.4:
        return False
    return True


def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
