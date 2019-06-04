import yake


def transcriptsToKeywords(transcripts, channels, ARGS):
    # 1. Create the extractor object while specifying parameters

    custom_extractor = yake.KeywordExtractor(
        lan=ARGS.language,
        n=ARGS.n,  # <-- How many words can a keyword consist of?
        dedupLim=0.4,
        dedupFunc='seqm',
        windowsSize=1,
        top=ARGS.top,  # <-- How many of top relevant keywords are returned?
        features=None)

    # 1. For each row in transcripts, extract keywords
    keywords = []

    if ARGS.use_subset:  # <-- for testing purposes
        print("Producing keywords for the first", ARGS.subsetSize,
              "transcripts. (--use_subset True --subsetSize", ARGS.subsetSize, ")")

        for idx in range(ARGS.subsetSize):
            print("Extracting keywords from transcript no. ", idx, " from channel: ", channels[idx])
            resulting_keywords = custom_extractor.extract_keywords(transcripts[idx])
            keywords.append(resulting_keywords)
    else:
        # print("Producing keywords for all transcripts found in csv file. ")
        for idx in range(len(transcripts)):
            # print("Extracting keywords from transcript no. ", idx, " from channel: ", channels[idx])
            resulting_keywords = custom_extractor.extract_keywords(transcripts[idx])
            keywords.append(resulting_keywords)

    # 2. For each transcript, sort keywords based on relevance score
    # print(keywords[0])
    # for keyword_result in keywords:
    # 	keyword_result.sort(key = lambda tup: tup[1], reverse = True)
    # print(keywords[0])
    # quit()
    # IS ALREADY SORTED

    # 3. Print the result of the first extraction
    return keywords
