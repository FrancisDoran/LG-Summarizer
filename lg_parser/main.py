from linkgrammar import Dictionary, ParseOptions, Sentence

#dpkg -L python3-link-grammar

"""
Note: The link grammar library returns iterators often
which need to be converted to lists to be printed.
"""

def print_linkage(lkg):
    print(lkg.diagram())
    print('Postscript:')
    print(lkg.postscript())
    print('---')

di = Dictionary('en')
#print(di.__class__.__name__) #Dictionary

po = ParseOptions()
#print(po.__class__.__name__) #ParseOptions

sent = Sentence("This is a simple sentence", di, po)
#print(sent.__class__.__name__) #Sentence

linkages = sent.parse()
#print(linkages.__class__.__name__) #sentence_parse

"""
first_linkage = linkages.next()
print(list(first_linkage.words()))
"""

for link in linkages:
    print("---------------------------")
    print(f"{list(link.links())}")
    print(f"Words: {list(link.words())}")
    print(f"Linkage diagram: {link.diagram()}")

