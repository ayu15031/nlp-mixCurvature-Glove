with open("data/book.txt", "r") as handle:
    txt = handle.readlines()

txt = " ".join(txt).replace("\n", "")

with open("data/book_clean.txt", "w") as handle:
    handle.write(txt)