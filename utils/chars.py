# -*- coding: utf-8 -*-

from __future__ import division
from sys import exit
from checker import *

import unicodedata as ud
import numpy as np


def strip_accents(s):
    """
    Dealing with some specific chars, e.g., convert Ç to "C + ̧"
    """
    u = unicode(s, 'utf-8')
    u_new = ''.join(c for c in ud.normalize('NFKD', u) if ud.category(c)!='Mn')
    return u_new.encode('utf-8')


def correct_unicode(string):
    string = strip_accents(string)
    string = re.sub("\xc2\xa0", " ", string).strip()
    string = re.sub("\xe2\x80\x93", "-", string).strip()
    #string = re.sub(ur'[\u0300-\u036F]', "", string)
    string = re.sub("â€š", ",", string)
    string = re.sub("â€¦", "...", string)
    #string = re.sub("[Â·ãƒ»]", ".", string)
    string = re.sub("Ë†", "^", string)
    string = re.sub("Ëœ", "~", string)
    string = re.sub("â€¹", "<", string)
    string = re.sub("â€º", ">", string)
    #string = re.sub("[â€˜â€™Â´`]", "'", string)
    #string = re.sub("[â€œâ€Â«Â»]", "\"", string)
    #string = re.sub("[â€¢â€ â€¡]", "", string)
    #string = re.sub("[â€â€‘â€“â€”]", "-", string)
    string = re.sub(ur'[\u2E00-\uFFFF]', "", string)
    string = re.sub("\\s+", " ", string).strip()
    return string


def simple_normalize(string):
    string = correct_unicode(string)
    # Citations
    string = re.sub("\[(nb ?)?\d+\]", "", string)
    string = re.sub("\*+$", "", string)
    # Year in parenthesis
    string = re.sub("\(\d* ?-? ?\d*\)", "", string)
    string = re.sub("^\"(.*)\"$", "", string)
    return string


def full_normalize(string):
    #print "an: ", string
    string = simple_normalize(string)
    # Remove trailing info in brackets
    string = re.sub("\[[^\]]*\]", "", string)
    # Remove most unicode characters in other languages
    string = re.sub(ur'[\u007F-\uFFFF]', "", string.strip())
    # Remove trailing info in parenthesis
    string = re.sub("\([^)]*\)$", "", string.strip())
    string = final_normalize(string)
    # Get rid of question marks
    string = re.sub("\?", "", string).strip()
    # Get rid of trailing colons (usually occur in column titles)
    string = re.sub("\:$", " ", string).strip()
    # Get rid of slashes
    string = re.sub(r"/", " ", string).strip()
    string = re.sub(r"\\", " ", string).strip()
    # Replace colon, slash, and dash with space
    # Note: need better replacement for this when parsing time
    string = re.sub(r"\:", " ", string).strip()
    string = re.sub("/", " ", string).strip()
    string = re.sub("-", " ", string).strip()
    # Convert empty strings to UNK
    # Important to do this last or near last
    if not string:
        string = "UNK"
    return string


def final_normalize(string):
    # Remove leading and trailing whitespace
    string = re.sub("\\s+", " ", string).strip()
    # Convert entirely to lowercase
    string = string.lower()
    # Get rid of strangely escaped newline characters
    string = re.sub("\\\\n", " ", string).strip()
    # Get rid of quotation marks
    string = re.sub(r"\"", "", string).strip()
    string = re.sub(r"\'", "", string).strip()
    string = re.sub(r"`", "", string).strip()
    # Get rid of *
    string = re.sub("\*", "", string).strip()
    return string
