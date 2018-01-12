#/usr/bin/python3

# https://gist.github.com/YoungxHelsinki/1c46b05cd16db4ee249c59717386aeb2

from concurrent.futures import ThreadPoolExecutor
import subprocess
import re

paniikki = [
    "befunge","bit","bogo","brainfuck","deadfish","emo","entropy","false","fractran","fugue","glass","haifu","headache","intercal","malbolge","numberwang","ook","piet","regexpl","remorse","rename","shakespeare","smith","smurf","spaghetti","thue","unlambda","wake","whenever","whitespace","zombie"
]
def get_luokka(nimi):
    try:
        return nimi, subprocess.check_output(["ssh", nimi, "who --count"], timeout=1, stderr=subprocess.STDOUT).decode("utf-8").rstrip("\n")
    except subprocess.TimeoutExpired:
        return nimi, "computer doesn't answer"
    except subprocess.CalledProcessError as e:
        return nimi, "error: %s" % e.output.decode("utf-8").rstrip("\n")

p = re.compile('# users=(\d+)')
executor = ThreadPoolExecutor(max_workers=8)

paniikki_stats = [(nimi, tulos) for nimi, tulos in executor.map(get_luokka, paniikki)]
paniikki_clean = [(nimi, p.search(users).groups()[0]) for nimi, users in paniikki_stats if p.search(users)]
print(sorted(paniikki_clean, key=lambda x: x[1])[0][0])
