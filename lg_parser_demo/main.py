from linkgrammar import Dictionary, Sentence, ParseOptions

def desc(lkg):
    print(lkg.diagram())
    print('Postscript:')
    print(lkg.postscript())
    print('---')

dict = Dictionary('en')

po = ParseOptions(min_null_count=0, max_null_count=999)

sent = Sentence("This is a simple sentence", dict, po)

linkages = sent.parse()

for link in linkages:
    desc(link)

