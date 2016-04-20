# -*- coding: utf-8 -*-
""" emoticon recognition via patterns.  tested on english-language twitter, but
probably works for other social media dialects. """

__author__ = "Brendan O'Connor (anyall.org, brenocon@gmail.com)"
__version__= "april 2009"

#from __future__ import print_function
import re,sys

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')

NormalEyes = r'[:=]'
Wink = r'[;]'

NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...

HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned

Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)

Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
Other_RE = mycompile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths )

Emoticon = (
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea + 
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"|"+HappyMouths+")"
)
Emoticon_RE = mycompile(Emoticon)

#SIMPLICACIo DEL VOCABULARI #

HTTP_RE = re.compile(r'''https?://\S+''', re.U)  #  <HTTP>
ARROBA_RE =re.compile (r'''@\S+''', re.U)  #  <ARROBA>
ALMO_RE =re.compile (r'''#\S+''', re.U)  #  <ALMO>
WEB_RE =re.compile (r'''www\S+''', re.U)  #  <WEB>
NUM_RE =re.compile (r'''\d+([,.]?\d)*''', re.U)  #  <NUM>
#EXCLAMA_RE = re.compile (r'''[¡]+''', re.U)  #  <EXCLAMA>
#PREGUNTA_RE = re.compile (r'''[?¿]+''', re.U)  #  <PREGUNTA>
#SIMBOL_RE =re.compile (r'''[^A-Za-zÁÉÍÓÚáéíóúüÜ0-9Ññ <>,.]+''', re.U)  #  <SIMBOL>

#Emoticon_RE = "|".join([Happy_RE,Sad_RE,Wink_RE,Tongue_RE,Other_RE])
#Emoticon_RE = mycompile(Emoticon_RE)

def analyze_tweet_original(text):
  h= Happy_RE.search(text)
  s= Sad_RE.search(text)
  if h and s: return "BOTH_HS"
  if h: return "HAPPY"
  if s: return "SAD"
  return "NA"


def analyze_tweet(text, http=True, emo=True, arroba=True, almo=True, web=True):
  if http:
    text= HTTP_RE.sub("<HTTP>", text )  # Es passa  primer perqu no detecte :// com emoticon
  if emo:
    text = Happy_RE.sub("<HAPPY>", text)
    text = Sad_RE.sub("<SAD>", text)
    text = Wink_RE.sub("<WINK>", text)
    text = Tongue_RE.sub("<TONGUE>", text)
    text= Other_RE.sub("<EMOTICON>", text)
  if arroba:
    text=ARROBA_RE.sub("<ARROBA>", text)
  if almo:
    text=ALMO_RE.sub("<ALMO>", text)
  if web:
    text=WEB_RE.sub("<WEB>", text)
#text10=NUM_RE.sub("<NUM>", text9)
#  text11=SIMBOL_RE.sub("<SIMBOL>", text10)
  #text11=EXCLAMA_RE.sub("<EXCLAMA>", text10)
#  text12=PREGUNTA_RE.sub("<PREGUNTA>", text11)
  
  return text
  
  
  # more complex & harder, so disabled for now
  #w= Wink_RE.search(text)
  #t= Tongue_RE.search(text)
  #a= Other_RE.search(text)
  #h,w,s,t,a = [bool(x) for x in [h,w,s,t,a]]
  #if sum([h,w,s,t,a])>1: return "MULTIPLE"
  #if sum([h,w,s,t,a])==1:
  #  if h: return "HAPPY"
  #  if s: return "SAD"
  #  if w: return "WINK"
  #  if a: return "OTHER"
  #  if t: return "TONGUE"
  #return "NA"

if __name__=='__main__':
  for line in sys.stdin:
    ##import sane_re
    ##sane_re._S(line[:-1]).show_match(Emoticon_RE, numbers=False)
    ##print "=="
    print (analyze_tweet(line.strip()), line.strip())
    ##print analyze_tweet(line.strip())
    ##print "----"
