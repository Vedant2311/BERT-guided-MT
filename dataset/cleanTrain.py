def readFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def filterDirtyPairs(token_list):
    return [(eng, nep) for eng, nep in token_list if (eng.strip() != nep.strip()) and (eng.strip() != '' and nep.strip() != '')]

def returnDirtyPairs(token_list):   
    return [(eng, nep) for eng, nep in token_list if (eng.strip() == nep.strip()) or (eng.strip() == '' or nep.strip() == '')]

def writeFile(sentences, file_path):
   with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(sentences))

def main():
    # Path to the "raw" training dataset (i.e. not tokenized)
    trainNepaliRawDirty = "train_raw/train_dirty.ne_NP"
    trainEnglishRawDirty = "train_raw/train_dirty.en_XX"

    # Path to the filtered "raw" training dataset
    trainNepaliRawClean = "train_raw/train.ne_NP"
    trainEnglishRawClean = "train_raw/train.en_XX"

    # Path to the removed dirty training pairs
    trainNepaliRawRemoved = "train_raw/removed/train.ne_NP"
    trainEnglishRawRemoved = "train_raw/removed/train.en_XX"

    # Replace 'english_file.txt' and 'nepali_file.txt' with your file paths
    english_sentences = readFile(trainEnglishRawDirty)
    nepali_sentences = readFile(trainNepaliRawDirty)

    # Ensure both files have the same number of lines
    if len(english_sentences) != len(nepali_sentences):
        print("Error: The number of lines in the two files doesn't match.")
        return

    # Create a list of tuples with (English sentence, Corresponding Nepali sentence)
    combinedSentences = list(zip(english_sentences, nepali_sentences))
    print("The original set length", len(combinedSentences))

    # Remove any repetitions over the original train dataset
    uniqueCombinedSentences = list(set(combinedSentences))
    print("The unique elements among them", len(uniqueCombinedSentences))

    # The documentations consist of many instances where the Nepali sentence and the English sentence are essentially the same
    # Those instances would usually be technical jargons. We basically have to make sure that we don't use those pairs for fine-tuning
    filteredCombinedSentences = filterDirtyPairs(uniqueCombinedSentences)
    print("The dirty pairs filtered", len(uniqueCombinedSentences) - len(filteredCombinedSentences))
    print("The final size of the training data", len(filteredCombinedSentences))

    # Separate out the English and the Nepali sentences and then write them to separate files
    filteredEnglishSentences = [eng.strip() for (eng, _) in filteredCombinedSentences]
    filteredNepaliSentences = [nep.strip() for (_, nep) in filteredCombinedSentences]

    # Writing these filtered sentences into a final output file
    writeFile(filteredEnglishSentences, trainEnglishRawClean)
    writeFile(filteredNepaliSentences, trainNepaliRawClean)

    # Saving the dirty pairs in a separate file
    dirtyCombinedSentences = returnDirtyPairs(uniqueCombinedSentences)

    # Separate out the English and the Nepali sentences and then write them to separate files
    filteredEnglishSentencesDirty = [eng.strip() for (eng, _) in dirtyCombinedSentences]
    filteredNepaliSentencesDirty = [nep.strip() for (_, nep) in dirtyCombinedSentences]

    # Writing these removed sentences into a final output file
    writeFile(filteredEnglishSentencesDirty, trainEnglishRawRemoved)
    writeFile(filteredNepaliSentencesDirty, trainNepaliRawRemoved)

if __name__ == "__main__":
    main()
