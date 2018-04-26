import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import multiprocessing as mp
import re
from collections import Counter

num_workers = mp.cpu_count()

def multithread(func, data, workers = num_workers):
    chunks = np.array_split(np.array(data),workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, chunks)
    return list(res)

def multiprocess(func, data, workers = num_workers):
    chunks = np.array_split(np.array(data),workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func,chunks)
    return list(res)

def convert_to_text_from_directory(DATA_DIR,OUT_DIR):
    for name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, name)
        filename, file_extension = os.path.splitext(name)
        output="{}/{}.txt".format(OUT_DIR,filename)

        if file_extension == ".pdf":
            os.system(("pdftotext %s %s") %(path,output))

        elif file_extension == ".docx":
            doc = docx.Document(path)
            fullText = []
            for para in doc.paragraphs:
                txt = para.text.encode('utf_8', 'ignore')
                fullText.append(txt.decode("utf-8"))

            with open(output, "w") as text_file:
                text_file.write("\n\n".join(fullText))

        elif file_extension == ".pptx":
            print(path)
            f = open(path, "r+b")
            presentation = Presentation(f)
            print(presentation)
            presentation = Presentation(path)
            print(presentation)
            fullText = []

            for slide in presentation.slides:
                for shape in slide.shapes:
                    if not shape.has_text_frame:
                        continue
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            fullText.append(run.text)

                with open(output, "w") as text_file:
                    text_file.write("\n\n".join(fullText))

def to_tsv(data,filepath):
    pd.DataFrame(data).to_csv(filepath,sep="\t",index=False,header=False)


def load_embedding_model(path):
    f = open(path,encoding="utf8")
    model = {}
    for line in f.readlines():
        splitLine = line.split(" ")
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    return model


def get_word_counts(corpus):
    words = re.sub("[^\w]", " ",  corpus).split()
    counts = Counter(words)
    return counts
