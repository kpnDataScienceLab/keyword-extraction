from models.key2vec import key2vec
import pandas as pd

doc_id = 10

# get transcript data
transcripts_fname = '../aligned_epg_transcriptions_npo1_npo2.csv'
transcripts = pd.read_csv(transcripts_fname)

# get description data
descriptions_fname = '../program_descriptions.csv'
descriptions = pd.read_csv(descriptions_fname)

# get text and tv show of the current tv show
text = transcripts['text'][doc_id]
tv_show = transcripts['prg.dap_program_title'][doc_id]

# get the description of that tv show
show_description = descriptions.loc[descriptions['dap_program_title'] == tv_show, 'dap_description_long']

if len(show_description) == 0:
    topic = tv_show
else:
    topic = show_description.iloc[0]

keywords = key2vec(text, topic, n=5)
