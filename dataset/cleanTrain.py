from tqdm import tqdm
from collections import defaultdict

# Filtering parameters
SHORT_THRESHOLD = 5
ENGLISH_LENIENCY_THRESHOLD = 0

def readFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def filterDirtyPairs(token_list):
    print("Cleaning the dataset...")
    clean_pairs = []
    dirty_pairs = []
    length_freq = defaultdict(int)
    for eng, nep in tqdm(token_list):
        length_freq[len(eng)] += 1
        length_freq[len(nep)] += 1
        if eng.strip() == nep.strip():
            dirty_pairs.append((eng, nep))
            continue

        # Remove sentences that are too short
        if len(eng) <= SHORT_THRESHOLD or len(nep) <= SHORT_THRESHOLD:
            dirty_pairs.append((eng, nep))
            continue
        
        # Remove the sentences that contain any english letters A-Z and a-z
        count = 0
        for c in nep:
            if ord('a') <= ord(c) <= ord('z') or ord('A') <= ord(c) <= ord('Z'):
                count += 1
            if count > ENGLISH_LENIENCY_THRESHOLD:
                dirty_pairs.append((eng, nep))
                break
        else:
            clean_pairs.append((eng, nep))
            
    # print("Top 10 most frequent lengths:")
    # for length, freq in sorted(length_freq.items(), key=lambda x: x[1], reverse=True)[:20]:
    #     print(f"{length}: {freq} ({freq/len(token_list)*2*100:.2f}%)")

    # for i in range(1, 8):
    #     freq = length_freq[i]
    #     print(f"{i}: {freq} ({freq/len(token_list)*2*100:.2f}%)")

    print(f"Cleaned out {len(dirty_pairs)} dirty pairs ({len(dirty_pairs)/len(token_list)*100:.2f}% removed)")
    return clean_pairs, dirty_pairs

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
    print("The original set length:", len(combinedSentences))

    # Remove any repetitions over the original train dataset
    uniqueCombinedSentences = list(set(combinedSentences))
    print("The unique elements among them:", len(uniqueCombinedSentences))

    # The documentations consist of many instances where the Nepali sentence and the English sentence are essentially the same
    # Those instances would usually be technical jargons. We basically have to make sure that we don't use those pairs for fine-tuning
    filteredCombinedSentences, dirtyCombinedSentences = filterDirtyPairs(uniqueCombinedSentences)
    print("The final size of the training data:", len(filteredCombinedSentences), f"({len(filteredCombinedSentences)/len(combinedSentences)*100:.2f}% of the original dataset)")

    # Separate out the English and the Nepali sentences and then write them to separate files
    filteredEnglishSentences = [eng.strip() for (eng, _) in filteredCombinedSentences]
    filteredNepaliSentences = [nep.strip() for (_, nep) in filteredCombinedSentences]

    # Writing these filtered sentences into a final output file
    writeFile(filteredEnglishSentences, trainEnglishRawClean)
    writeFile(filteredNepaliSentences, trainNepaliRawClean)

    # Saving the dirty pairs in a separate file
    # Separate out the English and the Nepali sentences and then write them to separate files
    filteredEnglishSentencesDirty = [eng.strip() for (eng, _) in dirtyCombinedSentences]
    filteredNepaliSentencesDirty = [nep.strip() for (_, nep) in dirtyCombinedSentences]

    # Writing these removed sentences into a final output file
    writeFile(filteredEnglishSentencesDirty, trainEnglishRawRemoved)
    writeFile(filteredNepaliSentencesDirty, trainNepaliRawRemoved)

if __name__ == "__main__":
    main()
