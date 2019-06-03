import pandas as pd
import nltk

def csvToTranscripts(filename = 'aligned_epg_transcriptions_npo1_npo2.csv'):

	data = pd.read_csv(filename)

	# get list of texts
	texts = data['text']
	channels = data['channel']

	return texts, channels

def keywordsTocsv(keywords_list, outputName = "tableOfKeywords"):
	
	# print(keywords_list[0])

	output_csv = []
	output_as_strings = []

	for keyword_results in keywords_list:
		row = []
		row_as_string = ""
		for keyword in keyword_results:
			row.append(keyword[0])
			row_as_string += keyword[0] + " -- "
		row_as_string = row_as_string[:-4]
		output_as_strings.append(row_as_string)
		output_csv.append(row)

	dataframe = pd.DataFrame(output_csv)
	dataframe = dataframe.add_prefix("Keyword:")
	dataframe.to_csv(outputName + '.csv')

	stringDataframe = pd.DataFrame(output_as_strings, columns = ["Keywords Extracted"])
	stringDataframe.to_csv(outputName + "_stringFormat.csv")
