## The data corresponding to GNOME, KDE4, and Ubuntu documentations can be found in the OPUS sub-folder under train_raw
## We made use of Flores v1's scripts to download this data for us. Ref: https://github.com/facebookresearch/flores/tree/main/previous_releases/floresv1
nepaliGNOMERaw = "train_raw/OPUS/GNOME.en-ne.ne"
englishGNOMERaw = "train_raw/OPUS/GNOME.en-ne.en"
nepaliKDE4Raw = "train_raw/OPUS/KDE4.en-ne.ne"
englishKDE4Raw = "train_raw/OPUS/KDE4.en-ne.en"
nepaliUbuntuRaw = "train_raw/OPUS/Ubuntu.en-ne.ne"
englishUbuntuRaw = "train_raw/OPUS/Ubuntu.en-ne.en"

## Somehow the scripts of Flores were not working to get the data corresponding to the Bible corpus
## We used the official github repo for the Bible corpus to achieve that - https://github.com/christos-c/bible-corpus
## The raw training data corresponding to the Bible corpus can be found in the sub-directory Bible
nepaliBibleRaw = "train_raw/Bible/Bible.en-ne.ne"
englishBibleRaw = "train_raw/Bible/Bible.en-ne.en"

def readFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def writeFile(sentences, file_path):
   with open(file_path, 'w', encoding='utf-8') as file:
        for s in sentences:
            file.write(s.strip() + "\n")

nepali_sentences_GNOME = readFile(nepaliGNOMERaw)
english_sentences_GNOME = readFile(englishGNOMERaw)

nepali_sentences_KDE4 = readFile(nepaliKDE4Raw)
english_sentences_KDE4 = readFile(englishKDE4Raw)

nepali_sentences_Ubuntu = readFile(nepaliUbuntuRaw)
english_sentences_Ubuntu = readFile(englishUbuntuRaw)

nepali_sentences_Bible = readFile(nepaliBibleRaw)
english_sentences_Bible = readFile(englishBibleRaw)

nepali_sentences = nepali_sentences_GNOME + nepali_sentences_KDE4 + nepali_sentences_Ubuntu + nepali_sentences_Bible
english_sentences = english_sentences_GNOME + english_sentences_KDE4 + english_sentences_Ubuntu + english_sentences_Bible

# Ensure both files have the same number of lines
if len(english_sentences) != len(nepali_sentences):
    print("Error: The number of lines in the two files doesn't match.")
else:
    # Strip the English and Nepali sentences and write them in separate files
    nepali_sentences_stripped = [s.strip() for s in nepali_sentences]
    english_sentences_stripped = [s.strip() for s in english_sentences]
    
    writeFile(nepali_sentences, "train_raw/train_dirty.ne_NP")
    writeFile(english_sentences, "train_raw/train_dirty.en_XX")

