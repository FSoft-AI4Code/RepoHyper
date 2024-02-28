from codebleu import calc_codebleu

def calc_metrics(hypotheses, references):
    result = calc_codebleu(hypotheses, references, lang="python")