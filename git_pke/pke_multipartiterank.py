import pke 
import nltk
import string
from pke import compute_document_frequency

topic_testing = "A B-52 bomber crashed and burst into flames early today on a runway while practicing \
				\"touch-and-go\" landings at K.I. Sawyer Air Force Base , officials said . All eight  \
				crew members survived . The plane , normally equipped to carry nuclear bombs , crashed \
				about 1:15 a.m. , said Lt. Naomi Siegal , a spokeswoman at the Strategic Air Command \
				installation . No weapons were aboard , said Lt. Col. George Peck , a spokesman for SAC \
				headquarters in Omaha , Neb. . The crew was practicing landings after a seven-hour training \
				flight when it crashed during one of its touch-and-go approaches , Peck said . During \
				such maneuvers , landing gears touch the ground but the plane doesn't land . All three \
				sections of the plane burned on impact , said Senior Airman Tim Sanders , a base spokesman \
				. The crew members crawled or were helped out of the front section of the aircraft , he said . \
				They were taken to Marquette General and base hospitals . Members of the crew suffered broken \
				bones , but no one was burned , said Capt. Paul Bicking , another Sawyer spokesman . \
				Senior Airman Tim Sanders , another base spokesman , said those aboard were Capt. \
				Mark Hartney , 29 , an aircraft commander from Mulberry , Fla. ; 1st Lt. Michael S. \
				Debruzzi , 26 , a pilot from New Brighton , Minn. ; Capt. Anthony D. Phillips , 28 , \
				a radar navigator from Folkston , Ga. ; 1st Lt. James W. Herrmann , 30 , a navigator \
				from Sharpsville , Pa. ; 1st Lt. Daniel McCarrick , 25 , an electronic warfare officer from \
				Succasunna , N.J. ; Airman 1st Class , Joseph A. Vallie , 20 , a gunner from Stephenson , \
				Mich. ; Maj. William R. Kroeger , 52 , an instructor pilot from Fountain Hills , Ariz. ; \
				and 1st Lt. Gregory C. Smith , 26 , an upgrade pilot from Henning , Minn. . All were based \
				at Sawyer . Ann Parent , a spokeswoman for Marquette General Hospital , said Hartney and \
				Debruzzi were in fair condition , Phillips and Vallie were in stable condition , McCarrick\
				 was in satisfactory condition and Kroeger was in serious condition . Herrmann and Smith \
				 were listed in stable condition at the base hospital , said Technical Sgt. Anita Bailey . \
				 Hartney was the aircraft commander , but Debruzzi also was qualified to fly the plane , \
				 Bailey said . She did not know who was at the controls at the time of the crash . \"We are \
				 counting our blessings ,\" Bailey said . \"You can put parts of a plane back together , but \
				 you can not put people back together .\" The accident was classified as the most serious kind , \
				 and all aircraft exercises at Sawyer were canceled even though runways other than the one \
				 where the crash occurred remained open , Bailey said . Peck said a board of officers will \
				 investigate the accident , adding weather did not appear to be a factor in the crash . \
				 Peck said it was not unusual for B-52 training missions to be out at that hour . \
				 \"Crews have to be trained to fly at any time of the day or night in any weather ,\" he said . \
				 The eight-engine B-52 , which was deployed in the early 1950s , is the military's biggest \
				 bomber , with a wingspan of 185 feet and a maximum takeoff weight of 488,000 pounds . The \
				 last B-52 was commissioned in 1962 . In other accidents involving B-52s , a bomber was \
				 damaged when a pilot aborted a takeoff and overshot a runway at Castle Air Force Base in \
				 central California on Feb. 11 . No one was injured . A B-52 bomber with radar problems \
				 crashed in Arizona'a Monument Valley in October 1984 , killing two crew members , after \
				 its wings clipped a mesa . The Air Force has had more trouble recently with the B-52's \
				 successor , the B-1B bomber . Although smaller than the B-52 , the B-1B can fly at \
				 supersonic speeds and carry more bombs . Four B-1Bs have crashed in the four years the \
				 plane has been flying , including two nine days apart in November . One of the $ 280 \
				 million B-1Bs was destroyed after smashing onto a runway at Ellsworth Air Force Base , \
				 S.D. , during a training flight on Nov. 17 . On Nov. 8 , a B-1B crashed and burned in a\
				  field near Dyess Air Force Base , Texas . No one was killed in either crash and \
				  investigators have not disclosed what caused the accidents . Six crewmen died and 10 \
				  were injured Oct. 11 when an Air Force tanker en route from K.I. Sawyer crashed at \
				  Wurtsmith Air Force Base near Oscoda . The Air Force's investigation of the crash is \
				  incomplete . Wurtsmith and Sawyer are Michigan's two SAC bases ."



def returnKeywords(topNkeyphrases):
	output = []
	for phrase in topNkeyphrases:
		output.append(phrase[0])
	return output

# MultipartiteRank
def pke_multipartiteRank(text, arguments, n = 5, language = 'dutch'):

	multiPartiteRank_extractor = pke.unsupervised.MultipartiteRank()
	parser_language = 'nl' if language == 'dutch' else 'en'
	
	POS = {'NOUN', 'PROPN', 'ADJ'}

	multiPartiteRank_extractor.load_document(input = text, language = parser_language)
	stoplist = list(string.punctuation)
	stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
	stoplist += nltk.corpus.stopwords.words(language)
	multiPartiteRank_extractor.candidate_selection(pos = POS,
												stoplist = stoplist)

	alpha = float(arguments[0])
	threshold = float(arguments[1])
	method = arguments[2]
	multiPartiteRank_extractor.candidate_weighting(alpha = alpha,
												threshold = threshold,
												method = method)
	keyphrases = multiPartiteRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)


# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

def test(text, arguments, k=5, lang = 'dutch'):
	if(lang == 'dutch'):
		return pke_multipartiteRank(text, arguments, n = k,  language = 'dutch')
	else:
		return pke_multipartiteRank(text, arguments, n = k,  language = 'english')


if __name__ == '__main__':

	transcript = ' '.join(readCleanTranscript("clean_transcripts_june11.txt", 0))

	print(transcript)
	result = test(transcript, 
				['1.1', '0.74', 'average'], 
				k = 20, 
				lang = 'dutch')
	print("Keyword predictions:\n", result)

