import pandas as pd
import datetime
import threading
from multiprocessing import Pipe, Pool, cpu_count

# load the data
in_text = ""
in_labels = ""
in_labels = pd.read_csv("../pattern/pattern.csv", sep=",")


def get_category(word):
    result = in_labels[in_labels.Term == word]
    if result.shape[0] != 0:
        return result.Tag.tolist()[0]
    return 'i'


def do_line(line):
    result_line = ""
    result_tags = ""
    if len(line) > 2:
        for word in line.split(' '):
            if (word != "," and "=" not in word and "<" not in word and "\"" not in word and "\'" not in word
                    and ">" not in word and "0" not in word and "1" not in word and "2" not in word and "3" not in word
                    and "4" not in word and "5" not in word and "6" not in word and "7" not in word and "8" not in word
                    and "9" not in word and ":" not in word and "@" not in word and "#" not in word and "(" not in word
                    and ")" not in word):
                result_line += word + " "
                result_tags += get_category(word) + " "
        result_line = result_line[:-1]
        result_tags = result_tags[:-1]
    return result_line, result_tags


def concat(line):
    mainf = line + "\n"
    tagf = label_text[new_text.index(line)] + "\n"


if __name__ == '__main__':
    print("Reading!")

    out_text = "final"
    out_labels = ""
    in_labels = ""
    in_labels = pd.read_csv("../pattern/pattern.csv", sep=",")

    with open("../test.txt", encoding="utf8") as f:
        in_text = f.readlines()
    f.close()
    # print(in_text[61])

    parent_conn, child_conn = Pipe()
    lck = threading.Lock()
    threads = []
    mainf = ""
    tagf = ""
    line_count = 0
    word_count = 0
    start_time = datetime.datetime.now()
    f = open("../output/text1.txt", 'w', encoding="utf8")
    g = open("../output/tags1.txt", 'w', encoding="utf8")

    # clean the input file
    print("Cleaning " + str(len(in_text)))

    in_text = list(dict.fromkeys(in_text))
    new_text = []
    label_text = []

    for line in in_text:
        if len(line) > 2:
            new_text.append(line)
    in_text = new_text
    new_text = []
    print("Cleaned " + str(len(in_text)))

    for line in range(len(in_text)):
        new_text.append("")
        label_text.append("")
    print("Pool deployment...")

    time1 = datetime.datetime.now()
    pool = Pool(cpu_count())
    results = pool.map(do_line, in_text)
    print("Pooling done, now reading results!")

    pool.close()
    pool.join()
    new_text = ""
    label_text = ""

    new_text = [result[0] for result in results]
    label_text = [result[1] for result in results]
    new_text = '\n'.join(new_text)
    label_text = '\n'.join(label_text)
    print("Cleaned " + str(len(new_text)))
    print("Done pooling: " + str(datetime.datetime.now().second - time1.second))

    line_count = 0
    print("OK...Outputting")

    print("Writing to files...")
    f.write(new_text)
    g.write(label_text)
    time_to_do = str(time1.second) + ":" + str(time1.microsecond)
    time_to_write = datetime.datetime.now() - time1
    tm_tr = str(time_to_write.seconds) + ":" + str(time_to_write.microseconds)

    print("Done writing to file, last calculation took: " + time_to_do + ", and write time for it took: " + tm_tr)

    f.close()
    g.close()
